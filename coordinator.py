from random import getrandbits, choice, randrange, randint
from string import ascii_lowercase 
from logging import getLogger, basicConfig, DEBUG
from enum import Enum
from time import sleep, time
from dataclasses import dataclass
from typing import Any, Tuple, List, Callable, Union
import threading
from mpi4py import MPI

Message = dict[str, Any]

class Tag(Enum):
    NOOP = 0
    TOKEN = 1
    ACK = 2

@dataclass
class Token:
    id: int = 0

class Coordinator:
    """
    A Coordinator class that creates a RING topology between MPI nodes.
    A single TOKEN circulates from node to node, granting access to the critical-section.
    The class does not provide an API to actually manage programs, but rather 
    implements an algorithm that strives to solve the problem of a vanishing token in a RING topology.
    The assumptions are:
        - the Coordinator processes are reliable (no outages),
        - and that only the channels between nodes are unreliable (messages can be lost);
    """

    def __init__(self, unreliable_channels: bool = True, unreliable_ack: bool = False):

        # MPI Variables
        self._comm: MPI.Intracomm = MPI.COMM_WORLD # mpi_context
        self.size: int = self._comm.Get_size()
        if self.size == 1:
            raise Exception(f"Ring intialized with a single node. Exiting...")
        self.rank: int = self._comm.Get_rank()
        # Lamport Logic
        self._lamport_clock: int = 0
        self._lamport_lock: threading.RLock = threading.RLock()
        # Token
        self._token: Token = None
        self._last_token_timestamp: Tuple[int, int] = (-1, -1)
        self._token_lock: threading.RLock = threading.RLock()
        self._token_unusable: bool = False
        self._wants_token: bool = False
        # Channels
        self.unreliable_channels: bool = unreliable_channels
        self.unreliable_ack: bool = unreliable_ack
        self.prev_node: int = (self.rank - 1) % self.size
        self.next_node: int = (self.rank + 1) % self.size
        # ACK        
        self._waiting_for_ack: bool = False
        self._ack_condition: threading.Condition = threading.Condition()
        # Logging
        basicConfig(level=DEBUG)
        self._logger = getLogger(__name__)
        self._logger.info(f"Coordinator[{self.rank=}] ready | prev: {self.prev_node} | next: {self.next_node} | size: {self.size=}")

        prefix_message: str = f"[{self.rank}] | "

        self._logger.info = self.logger_padding(" "*4+prefix_message)(self._logger.info)
        self._logger.error = self.logger_padding(" "*3+prefix_message)(self._logger.error)
        self._logger.debug = self.logger_padding(" "*3+prefix_message)(self._logger.debug)
        self._logger.warning = self.logger_padding(" "+prefix_message)(self._logger.warning)
        self._logger.critical = self.logger_padding(prefix_message)(self._logger.critical)

        # Root init
        if self.rank == 0:
            self._token: Token = Token()
            with self._token_lock:
                with self._lamport_lock:
                    self._last_token_timestamp = (self._token.id, self._lamport_clock)
                    self._wants_token: bool = False
            thread = threading.Thread(target=self._release_token, args=(True, ))
            thread.start()

    @staticmethod
    def logger_padding(message_prefix: str):
        def logger_decorator(logger_function: Callable):
            def logger_wrapper(arg):
                padding_str: str = '{prefix_message:{fill}{align}{width}} {main_message}'.format(
                    prefix_message=message_prefix,
                    fill=' ',
                    align='>',
                    width=0,
                    main_message=arg
                )
                logger_function(padding_str)
            return logger_wrapper
        return logger_decorator

    def send_message_forward(self, message: Message, tag: Union[Tag, int]) -> None:
        """
        A wrapper function to send a message to the next node in the ring, using _send_message().

            Args:
                message (dict[str, Any]): A dictionary representing the data to be send
                tag (Union[Tag, int]): either an integer value of the tag or an enum alias for the integer

            Returns: None
        """
        self._send_message(self.next_node, message, tag)

    def send_message_backward(self, message: Message, tag: Tag) -> None:
        """
        A wrapper function to send a message to the previous node in the ring, using _send_message().

            Args:
                message (dict[str, Any]): A dictionary representing the data to be send
                tag (Union[Tag, int]): either an integer value of the tag or an enum alias for the integer

            Returns: None
        """
        self._send_message(self.prev_node, message, tag)

    def _send_message(self, recipient: int, message: Message = {}, tag: Union[Tag, int] = Tag.NOOP) -> None:
        """
        A function to send a message to any node in the ring.

            Args:
                recipient (int): an MPI rank value of the target node
                message (dict[str, Any]): A dictionary representing the data to be send
                tag (Union[Tag, int]): either an integer value of the tag or an enum alias for the integer

            Returns: None
        """
        if self.rank == recipient:
            self._logger.error(f"sending a message to itself")
        
        if type(tag) is not int:
            tag = tag.value

        # Unreliable send channel 50% chance
        channel_failure: bool = False
        if self.unreliable_channels:
            channel_failure: bool = not getrandbits(1)
            if channel_failure:
                self._logger.warning(f"channel failure | link: ({self.rank})->({recipient}) | tag: ({tag})")
                return

        # Unreliable ACK 100% of the time
        if self.unreliable_ack:
            if tag == Tag.ACK.value:
                self._logger.warning(f"channel ACK failure | link: ({self.rank})->({recipient}) | tag: ({tag})")
                return

        # ===============------=======
        with self._lamport_lock:
            self._lamport_clock += 1
            data: Message = {
                'clock': self._lamport_clock,
                'recipient': recipient, # The MPI rank id of the designated node,
                                        # could be used if messages were to be 
                                        # send with the exclusion of some nodes (pass-through)
            }
        data: Message = message | data # Update the message with basic info | merges dicts
        
        self._logger.debug(f"sending a message to: ({recipient}) | tag: ({tag}) | msg: ({data})")
        
        self._comm.send(data, dest=recipient, tag=tag)

    def _deliver_message_thread(self, message: Message, tag: Tag) -> None:
        """
        A function invoked as a new thread in '_receive_message_thread()'.
        When a message is meant to be received by the node the function handles its processing.
        Message's Token cases:
            - Tag.TOKEN:
                1) sends ACK backwards
                2.a) if the node doesn't want nor have the token, it passes it forward
                2.b) if the node wants the token but does not have it, it aquires the token,
                    executes its critical section and releases the token forward, and waits for an ACK  
                2.c) if the node has a token and receives one, it updates the old token

            Args:
                message (dict[str, Any]): A dictionary representing the data received
                tag (Union[Tag, int]): either an integer value of the tag or an enum alias for the integer

            Returns: None
        """
        if tag == Tag.TOKEN.value:
            self._logger.debug(f"delivering a <TOKEN> message: ({message})")
            received_token: Token = message.get('token', None)
            received_token_clock: int = message['clock']
            if not received_token:
                self._logger.critical(f"received a Tag.TOKEN message, but the <TOKEN> is missing")
                return
            self._send_ack()

            # Stop regenerating tokens while waiting for Acks if token was received
            with self._ack_condition:
                self._waiting_for_ack = False
                self._ack_condition.notify()

            self._token_lock.acquire()
            if self._token: # and self._token_unusable?
                if received_token.id < self._token.id:
                    pass
                elif received_token.id > self._token.id:
                    self._logger.info(f"received an updated token: ({received_token})>({self._token})")
                    # Update the current token | useless token is removed
                    self._token = received_token
                    with self._lamport_lock:
                        self._last_token_timestamp = (received_token.id, self._lamport_clock)
                else:
                    pass
                self._token_lock.release()
            elif self._wants_token:
                self._lamport_lock.acquire()
                # Process token
                if self._last_token_timestamp[1] < received_token_clock:
                    # The last used token is older than the new token -> TOKEN SAFE (NO TOKEN DUPLICATION)
                    self._token_lock.release()
                    self._lamport_lock.release()
                    self._critical_section(received_token)
                    return
                elif self._last_token_timestamp[1] > received_token_clock:
                    # TODO: CHYBA TO WYKOMENTUJ!
                    # Duplicate token, <ACK> to prev_node was lost, and so a duplicate was generated
                    # let it pass through to be stopped at the node with the old token. 
                    self._logger.info(f"received a duplicate token: ({message['token']})")
                    # Forward token and wait for ACK 
                    
                    # Do not even forward # FIXME
                    self._token_lock.release()
                    self._lamport_lock.release()
                    # self.send_message_forward(message, Tag.TOKEN)
                    # self._receive_ack()
                elif self._last_token_timestamp[1] == received_token_clock:
                    self._logger.debug(f"Same timestamp clock: ({message['token']}) (SCENRAIO 3)") # FIXME remove
                    self._token_lock.release()
                    self._lamport_lock.release()
            else:
                self._token_lock.release()
                # Pass the token further
                self._lamport_lock.acquire()
                forward_mesage: bool = False
                if received_token_clock > self._last_token_timestamp[1]:
                    self._last_token_timestamp = (received_token.id, self._lamport_clock)
                    # Send only if new
                    self._logger.info(f"passing the token further: ({received_token})")
                    self._lamport_lock.release()
                if forward_mesage:
                    self.send_message_forward(message, Tag.TOKEN)
                    # Forward token and wait for ACK 
                    self._receive_ack()
            # self._logger.critical("finally releasing lock") # FIXME USUN

        elif tag == Tag.ACK.value:
            self._logger.debug(f"delivering an <ACK> messafge: ({message})")
            with self._ack_condition:
                self._waiting_for_ack = False
                self._ack_condition.notify()
        else:
            self._logger.critical(f"received a message with an unknown <TAG>: ({tag})")

    def _receive_message_thread(self) -> None:
        """
        An infinite loop that listens for new messages. 
        If the message is meant for the node, it starts a new thread to handle it ('_deliver_message_thread'),
        else it forwards the message in another thread ('send_message_forward').

            Args:
                None

            Returns: None
        """
        while True:
            status: MPI.Status = MPI.Status()
            received_data: Message = self._comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source: int = status.Get_source()
            tag: int = status.Get_tag()

            received_clock: int = received_data.get('clock', None)

            self._logger.info(f"received: {received_data}")

            if not received_clock or type(received_clock) is not int:
                self._logger.critical(f"received a message without a proper <CLOCK>")
                continue

            with self._lamport_lock:
                self._lamport_clock = max(received_clock, self._lamport_clock) + 1
           
            if received_data['recipient'] == self.rank:
                self._logger.debug(f"delivering a message with: ({received_data})")
                thread = threading.Thread(target=self._deliver_message_thread, args=(received_data, tag))
                thread.start()
            else:
                # Unused now, but could be used for some special messages
                self._logger.debug(f"forwarding a message with: ({received_data})")
                thread = threading.Thread(target=self.send_message_forward, args=(received_data, tag))
                thread.start()

    def _aquire_token(self, token: Token) -> None:
        """
        A token can be acquired when the Coordinator wants to acquire it and does not already have a token.
        When acquiring the token the Coordinator updates the token reference and changes the state
        of 'self._wants_token'.

            Args:
                token (Token): the new token to be acquired

            Returns: None
        """
        with self._token_lock:
            if not self._token and self._wants_token:
                self._logger.debug(f"aquiring token")
                self._token = token
                with self._lamport_lock:
                    # TODO: TUTAJ JAKIS WARUNEK ZE IF JESLI SA NOWSZE DANE!?
                    self._last_token_timestamp = (token.id, self._lamport_clock)
                self._wants_token = False
            else:
                self._logger.critical(f"wants to aquire the token but cannot do so")

    def _release_token(self, initWaitingPhase: bool = False) -> None:        
        """
        After using the token the Coordinator passes it forward and nulls the token reference.
        After releasing the token it waits for an ACK from the next node in '_receive_ack()'

            Args:
                None

            Returns: None
        """
        released_token: bool = False
        with self._token_lock:
            # If token is no longer needed
            # if self._token and not self._wants_token:
            if self._token:
                self._logger.debug(f"releasing token: ({self._token})")
                data = {
                    'token': self._token,
                }
                self._token = None
                self._token_unusable = False
                # Forward the token
                self.send_message_forward(message=data, tag=Tag.TOKEN)
                released_token = True
            else:
                self._logger.error(f"wants to release the token but cannot do so")
        # Wait for ACK
        if initWaitingPhase and released_token:
            self._receive_ack()

    def _critical_section(self, token: Token) -> None:
        self._logger.debug(f"preparing for the critical section")
        self._aquire_token(token)
        self._logger.debug(f"entering the critical section | ({self._token})")
        sleep(randint(2, 4)) # Time consuming token-related activites
        self._release_token(initWaitingPhase=True)

    def _regenerate_token(self) -> None:
        """
        Creates a new token in case it may have been lost and then releases it.

            Returns: None
        """
        with self._token_lock:
            new_token_id: int = self._last_token_timestamp[0] + 1
            self._token: Token = Token(new_token_id)
            with self._lamport_lock:
                self._last_token_timestamp: int = (new_token_id, self._lamport_clock)
            self._token_unusable = True
            self._logger.debug(f"regenerated a token: ({self._token})")

    def _send_ack(self) -> None:
        """
        Sends an <ACK> message to the previous node in the ring upon receiving a <TOKEN> message.

            Args:
                None

            Returns: None
        """
        self._logger.debug(f"sending ACK to: ({self.prev_node})")
        self.send_message_backward(message={}, tag=Tag.ACK)

    def _receive_ack(self) -> None:
        """
        The Coordinator waits for an ACK message for the
        designated amount of time, based on (allowed_timeout_seconds).
        Unless it receives the ACK in the given timespan, it invokes
        the '_regenerate_token()' method to create a new token, send it forward,
        and wait for a new ACK again.
        ACKs are not identified in any way -> for example ACK(NEW_TOKEN_ID).
        The program does not wait for all ACKs, but only for the latest one.
        This is because it does not matter if all TOKENS make it through the channel,
            but at least one of them.
        
        The problem (channel failure):
            The coordinator may not receive an ACK on time:
                - either the ack message was lost,
                - or the <TOKEN> message never made it to the next node;
            Because of this the Ring may end up with multiple tokens that have to be resolved later on.

            Returns: None
        """
        # Make sure to initialize ack waiting only once so that 
        # once the Ack is received in another thread or a token is
        # received in the meantime it is properly stopped
        with self._ack_condition:
            self._waiting_for_ack = True

        allowed_timeout_seconds: int = 5.0
        
        while True:
            self._logger.debug(f"preparing to wait for ACK from ({self.next_node})")
            sleep(3)

            allowed_timeout_timestamp = time() + allowed_timeout_seconds
            ack_received: bool = False
            with self._ack_condition:
                while self._waiting_for_ack and time() < allowed_timeout_timestamp and not ack_received:
                    ack_received: bool = self._ack_condition.wait(timeout=allowed_timeout_seconds)

            should_regenerate_token: bool = False
            with self._ack_condition:
                if self._waiting_for_ack and not ack_received:
                    self._logger.debug(f"did not receive ACK from ({self.next_node})")
                    should_regenerate_token = True
            if should_regenerate_token:
                self._regenerate_token()
                self._release_token(initWaitingPhase=False)
            else:
                with self._ack_condition:
                    self._waiting_for_ack = False
                if not ack_received:
                    self._logger.debug(f"stopped waiting for ACK from ({self.next_node}) -> received <TOKEN>")
                else:
                    self._logger.debug(f"received ACK from ({self.next_node})")
                break

    def _think_about_token_thread(self) -> None:
        """
        Each random interval, the Coordinator can decide whether it needs the token or not.
        Only if it does not have the token nor already wants it.

            Args:
                None

            Returns: None
        """
        while True:
            with self._token_lock:
                if not self._token_unusable and not self._wants_token:
                    # 50% chance of wants token
                    self._wants_token: bool = not getrandbits(1)
                    # == 0b0101
                    if self._wants_token:
                        self._logger.debug(f"decided to require the token")
                    else:
                        pass
                        # self._logger.debug(f"still does not need the token")
                else:
                    pass
                    # self._logger.debug(f"failed with deciding token")
            sleep(randint(5, 8))

    def ring_loop(self) -> None:
        """
        The main method of each Coordinator that sets up the ring
        and starts the necessary threads. 
            The listening thread - '_receive_message_thread()'
            and the simulation one - '_think_about_token_thread()'

            Args:
                None

            Returns: None
        """
        self._logger.debug(f"entered the ring loop")
        thread_funcs: dict[Callable, Tuple] = {
            self._receive_message_thread: (),
            self._think_about_token_thread: (),
        }

        thread_list: List[threading.Thread] = []
        for thread_func, args in thread_funcs.items():
            thread: threading.Thread = threading.Thread(target=thread_func, args=args)
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            # listen loop never finishes | sigint kills but maybe use class threading.Event to kill? # FIXME?
            thread.join()

    def __call__(self):
        self.ring_loop()

def main():
    coordinator: Coordinator = Coordinator()
    coordinator()

if __name__ == '__main__':
    import pathlib
    import sys

    assert sys.version_info >= (3, 10), "Script requires Python 3.10+."
    here = pathlib.Path(__file__).parent
    main()
