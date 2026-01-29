# k-way merge algorithm

Parallel k-way merge algorithm for merging k sorted lists with arbitrary sizes. Lists are divided into sublists that can be merged independently. Sublists are merged iteratively two at a time using a [bitonic network](https://en.wikipedia.org/wiki/Bitonic_sorter).

## Table of Contents

- [Implementation](#implementation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Performance Comparison](#performance-comparison)
- [License](#license)

## Implementation

### Generating splitters

Every list is divided into p sublists using p+1 splitters, where p is a multiple of the number of Streaming Multiprocessors (SM). For every list, the starting and ending splitters are the starting and ending positions of the list. A new set of splitters (one for every list) is created in such a way that the new sublists contain the n/p smallest numbers not yet partitioned, where n is the length of the merged list.

The correct cut-off point is calculated using parallel binary search. We choose a random number from the unpartitioned part of a random list. Every thread then uses binary search to count the number of unpartitioned elements less than the pivot and less than or equal to the pivot. The total count of smaller elements and smaller or equal elements is calculated using parallel reduction. If the target size of n/p elements is between the two counts, the correct pivot is found. Otherwise, we update the search boundaries for all lists and continue the search.

Once the correct pivot is found, we can generate the splitters by adjusting the number of elements equal to the pivot that we include in each sublist.

### Merging

Sets of sublists are merged in parallel. Two sublists are merged at a time (first and second, third and fourth, etc.) until there are no more sublists to merge.

Let T be the number of threads in a block. At first, T elements from each sublist are loaded into a buffer so they form a bitonic sequence. These elements are then sorted using the bitonic mergesort algorithm. First T elements from the buffer are copied to output and then replaced with T new elements from the sublist with the smallest next element. The buffer is sorted again, and the steps are repeated until the two sublists are fully processed.

## Requirements

To use this project, you need the following:

- An NVIDIA GPU that supports CUDA.
- The NVIDIA CUDA Toolkit.

## Usage

The code can be compiled with `nvcc -o k_way_merge k_way_merge.cu` and run with `./k_way_merge k sizes.dat elements.dat results.dat`. `k` is the number of lists and `sizes.dat` and `elements.dat` are binary files containing the list sizes and elements, respectively. The merged list will be saved to `results.dat`.

## Performance Comparison

`k_way_merge` compares the parallel k-way merge algorithm on a GPU to the simple iterative 2-way merge algorithm on the CPU.

`test.cu` is a helper script that can be compiled with `nvcc -o test test.cu` and run with `./test k max_n`. It generates `k` lists with arbitrary sizes between `1` and `max_n` and saves them on disk.

I generated 1024 lists with sizes of up to 200000. Fo testing, I used Intel Core i5-7300HQ and NVIDIA GeForce GTX 1050 Ti. The CPU merged the lists in about 10 s. The GPU divided the lists in 18 ms and merged the sublists in 769 ms.

<img width="1574" height="210" alt="test-results" src="https://github.com/user-attachments/assets/d0248eb5-e419-494b-89b8-bae379afd3ba" />

## License


