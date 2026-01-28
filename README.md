# k-way merge algorithm

Parallel k-way merge algorithm for merging k sorted lists with arbitrary sizes. Lists are divided into sublists that can be merged independently. Sublists are merged iteratively two at a time using a bitonic network.

## Table of Contents

- [Implementation](#implementation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Performance Comparison](#performance-comparison)
- [License](#license)

## Implementation

### Generating splitters

Every list is divided into p sublists using p+1 splitters, where p is a multiple of the number of Streaming Multiprocessors (SM). For every list, the starting and ending splitters are the starting and ending positions of the list. A new set of splitters (one for every list) is created in such a way that the new sublists contain the n/p smallest numbers not yet partitioned, where n is the length of the merged list.

The correct cut-off point is calculated using parallel binary search. We choose a random number from the unpartitioned part of a random list. Every thread then uses binary search to count the number of unpartitioned elements less than the pivot and less than or equal to the pivot. The total count of smaller elements and smaller or equal elements is calculated using parallel reduction. If the target size of n/p elements is between the two counts, the correct pivot is found. Otherwise, we update the search boundaries for all lists and continue the search.

Once the correct pivot is found, we can generate the splitters by adjusting the number of elements equal to the pivot that we include in each sublist.

### Merging

Sets of sublists are merged in parallel. Two sublists are merged at a time (first and second, third and fourth, etc.) until there are no more sublists to merge.

Let T be the number of threads in a block. At first, T elements from each sublist are loaded into a buffer so they form a bitonic sequence. These elements are then sorted using the bitonic mergesort algorithm. First, T elements from the buffer are copied to output and then replaced with T new elements from the sublist with the smallest next element. The buffer is sorted again, and the steps are repeated until the two sublists are fully processed.
