# XNNs - Cross Neural Networks

## 1 - Abstract

The XNN, or Cross Neural Network, is a novel form of neural network, which aims to embody
bio-analogous properties, which current models fundamentally lack, at the core of the architecture.

The intention of this is that innate deliberation mechanisms may arise in processing, rendering the
architecture a viable option for AGI, or artificial general intelligence.

## 2 - Contents

### 2.1 - Overview

XNNs are similar in form factor to Hopfield networks, but only superficially.

In an XNN, each node is connected to all nodes in the network, including itself, the connections
between nodes being one-way, allowing the connections to have different properties in different
directions. This makes XNN connections very easy to represent as simple adjacency matrices. Each
node contains a state at any given time, and these can be stored in a simple vector alongside the
matrix.

Node states and connection properties can consist solely of numerical activations and weights, but
certain proposals feature what are referred to as "atypical neural parameters" (ATNP), this being
anything beyond the aforementioned values.

The network processes information in "steps", where the state of each node is sent over its
connections to all other nodes, being transformed by the properties of said connections, after
which the values received by each node are compiled into the new state of the node for the next
step. If no ATNP is applied, this process should resemble what takes place in a typical feed
forward neural network when data moves from one layer to the next.

However, since there is no clear input or output section in such a model, data is to be processed
continuously, with input being the manual editing of the states of certain nodes, and output being
the reading of the states of certain nodes, as the network continues processing. This mode of
real-time continuous processing and interaction is referred to as asynchronous neural processing
(ASNP).

Since the absence of a connection can be represented by a connection weight of zero; since any
graph can be said to contain a subset of the connections of a similar graph where all nodes are
connected; and since any conceivable neural network must be some sort of graph; it can be said that
any conceivable neural network architecture can be replicated within an XNN.

Therefore, if any neural network architecture can achieve AGI-level cognition, it can be said that
an XNN is capable of this.

However, XNNs can go beyond the capabilities of other architectures in that their hyper-recurrent
and continuous nature allows for a near indefinite degree of reflective cognition, as well as
unique abilities to embody active and spontaneous initiative, whereas most AI models are purely
reactive systems.

In seeking to render XNNs as adaptive as possible to an innumerable variety of contexts, they
should be trained exclusively with reinforcement learning.

### 2.2 - Goals

The primary thing lacking is a reliable and well-tested reinforcement training methodology for
XNNs. Though there are proposals for such methods, they have yet to consistently demonstrate their
efficacy.

Among such methods are:

- Taking a sequence of XNN steps, and treating each of them as a layer in a traditional feed
forward network, with the caveat being that the connections between different layers are the same.
This may allow something akin to more traditional back propagation methods.

- Keeping "heat scores" on the connections as part of ATNP to indicate the degree of their recent
contributions to the result of the network, thus allowing them to be proportionately rewarded or
punished when feedback is delivered.

- Assigning activations passed through the network types, with certain types used for processing
and others used for training. This functionality is intended to mimic that of neurotransmitters.