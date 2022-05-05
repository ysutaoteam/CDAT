# CDAT Feature Extraction
We provide the code for the CDAT feature extraction part. Please follow the instructions below to use our code.

# Prerequisites
The code is tested on 64 bit Windows 10. You should also install Python 3.6 before running our code.

# 1) Establish the formal context.py
The input of this file is the spectrum. The spectrum is divided into several sub-regions after sliding windows. The pixel points in the sub-regions are used as objects and the quantized direction intervals as attributes. The output is a formal context table in the form of objects as columns and direction attributes as rows.

# 2) Format conversion.py
The purpose of this file is to convert the formal context to mat format for further processing.

# 3) Output the adjacency matrix as an AT graph.py
This file takes the formal context as input and outputs CDAT for visualization. CDAT uses attributes as nodes, and the values over the lines connecting two nodes are the weight information between the attributes.

# 4) Count the number of connected domains of AT.py
The input of this file is CDAT and the output is the number of connected components in each sub-region. The output table can be fed directly to the classifier.
