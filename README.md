## SensorChain 3.0
The latest iteration of SensorChain.


### Installation
Python 3 dependencies:
1. pycryptodome
2. pypubsub
3. jsonpickle

Once the necessary modules are installed, simply navigate to the root directory and run main.py.

This version makes use of multiprocessing to run multiple tests at once.
To run a certain dataset, uncomment the corresponding multiprocess expression in main.py.

WARNING: The datasets are very large; with enough time the computer running the code could slow down and begin stalling.
This is to be expected and is not an issue with the code. For a brief run use test 10.
