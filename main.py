
from coordinator import Coordinator

# from hanging_threads import start_monitoring
# start_monitoring(seconds_frozen=10, test_interval=100)

def main():
    coordinator = Coordinator(unreliable_channels=True, unreliable_ack=False)
    coordinator()
    

if __name__ == '__main__':
    main()
