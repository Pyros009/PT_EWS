import numpy as np
import logging
import logging.handlers
import time

from obspy import Trace, UTCDateTime, Stream
from obspy.clients.seedlink.easyseedlink import create_client
from obspy.clients.fdsn import Client as FDSNclient
from obspy.core.inventory import Inventory

from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
import seaborn as sns

from fetch import DataFetcher
from process import process_trace

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.NOTSET,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

### MAIN        
def main():
    # Create a multiprocessing queue for communication
    q = Queue()

    logging.info('Starting data fetching process')

    # source data
    url = 'rtserve.iris.washington.edu:18000'
    source = {"LX":{"MORF":["BHE","BHN","BHZ"]
                #,"GGNV":["BHN","BHZ","HHE","HHN","HHZ"]
                 }
            }

    # Create an instance of DataFetcher
    data_fetcher = DataFetcher()
    
    
    p1 = Process(target=data_fetcher.fetch_data, args=(q,url, source))
    p2 = Process(target=process_trace, args=(q,))
  
    
    try:
        p1.start()
        p2.start()
        
        p1.join()
        """
        logging.info("Waiting to receive...")
        while True:
            if not q.empty():
                item = q.get()  # Retrieve item from queue
                print(f"Received: {item}")
            else:
                # Log queue size to help diagnose if data is being sent
                time.sleep(0.1)  # Slight delay to avoid busy waiting
        """
        
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt. Exiting...")

    #finally:
        #p1.terminate()  # Terminate the child process
        #p1.join()       # Wait for it to finish
        #logging.info("Data fetching process terminated.")

if __name__ == "__main__":
    main()