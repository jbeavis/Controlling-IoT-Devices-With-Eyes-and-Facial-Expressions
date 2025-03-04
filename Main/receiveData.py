# This example came with the library
from pylsl import StreamInlet, resolve_byprop
import pandas as pd

def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_byprop("type", "EEG")

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        print(timestamp, sample)

    return timestamp, sample

# Figure out what a dataframe looks like
# When data is captured, put it in a dataframe
# Or just extract features then and there?
# When enough has been captured, analyse features
# Constantly shift dataframe left, adding new samples as they come in
# After x number of samples have come in, analyse dataframe again


if __name__ == "__main__":
    main()
