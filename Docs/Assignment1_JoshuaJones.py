### CSCI-B505 Assignment 1
### Joshua Jones          
### Eval perf of 3 sorting algos

from math import inf
import time as ti
import random as rand
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from plotly.io import write_image
import sys
import pytest


# Accepts list of integers and performs deterministic quick sort in place
def det_qs(nums:list) -> None:
    r = len(nums) - 1
    p = 0
    quicksort(nums, p, r, False)
    

# Accepts list of integers and performs random quick sort in place
def random_qs(nums:list) -> None:
    r = len(nums) - 1
    p = 0
    quicksort(nums, p, r, True)
    

# Accepts list of integers and performs quick sort in place
def quicksort(nums:list, p:int, r:int, rand_pivot:bool=False) -> None:
    if p < r:
        if rand_pivot:
            q = partition(nums, p, r)
        else:
            q = partition_random(nums, p, r)
        quicksort(nums,   p, q-1, rand_pivot)
        quicksort(nums, q+1,   r, rand_pivot)


# Performs in place partitioning to arrange all lesser values below or greater above index r
def partition(A:list, p:int, r:int) -> int:
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1


# Runs partitioning with randomized partition index
def partition_random(A:list, p:int, r:int) -> int:
    r = rand.randrange(p,r)
    A[r], A[p] = A[p], A[r]
    return partition(A, p, r)


# Accepts list of integers and performs merge sort in place
def merge_sort(nums:list, m:int, r:int) -> None:
    if m < r:
        q = (m + r) // 2            # approx middle index
        merge_sort(nums,  m,  q)    # sort first segment
        merge_sort(nums, q+1, r)    # sort second segment
        merge(nums, m, q, r)        # combine 


# Combine each sorted segment in ascending order
def merge(A:list, m:int, q:int, r:int) -> None:
    L = []
    R = []
    n1 = q - m + 1
    n2 = r - q

    # slice rather than copy via loop
    slice = A[m:m+n1]
    L = slice.copy()

    slice = A[q+1:q+1+n2]
    R = slice.copy()
    
    L.append(inf)
    R.append(inf)

    i = 0
    j = 0
    for k in range(m,r+1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1


# Generates cases # of evenly spaced test vectors pairs (sorted, randomized) between 0 and max_n
def generate_inputs(cases:int, max_n:int) -> dict:
    tests = {}
    interval = max_n // cases
    # generate cases number of test arrays with size interval
    for n in range(1, max_n+1, interval):
        test_sorted = [i for i in range(1, n+1)]
        test_random = test_sorted.copy()
        rand.shuffle(test_random)
        tests.update({n: (test_sorted, test_random)})
    return tests


# Wrapper to time the execution time of each test vector
def algo_timer(a_func, nums:list, is_input_sorted:bool, n:int) -> float:
    t1 = ti.time()
    # print(f'\n{a_func.__name__} called -',end='')

    # call function based on sort method
    if a_func.__name__ == 'merge_sort':
        a_func(nums,0,len(nums)-1)
    else:
        a_func(nums)

    t_delta = (ti.time()-t1)*1000
    return t_delta



@pytest.fixture()
def test_me():
    print('this runs before the sub')
    yield 'sub'
    print('this runs after the sub')


if __name__ == "__main__":

    sys.setrecursionlimit(15000)
    # print(f'Max Recursion Depth: {sys.getrecursionlimit()}')

    t_start = ti.time()

    # test input generation
    cases = generate_inputs(cases=50, max_n=1000)

    det_qs_sorted_t   = []
    det_qs_unsorted_t = []

    random_qs_sorted_t   = []
    random_qs_unsorted_t = []

    merge_sort_sorted_t   = []
    merge_sort_unsorted_t = []

    for n,(sorted,unsorted) in cases.items():

        # Deterministic Quick Sort
        det_qs_sorted_t.append(  algo_timer(det_qs,   sorted,  True, n))
        det_qs_unsorted_t.append(algo_timer(det_qs, unsorted, False, n))

        # Random Quick Sort
        random_qs_sorted_t.append(  algo_timer(random_qs,   sorted,  True, n))
        random_qs_unsorted_t.append(algo_timer(random_qs, unsorted, False, n))

        # Merge Sort
        merge_sort_sorted_t.append(  algo_timer(merge_sort,   sorted,  True, n))
        merge_sort_unsorted_t.append(algo_timer(merge_sort, unsorted, False, n))


    # Generate plots
    # fig = make_subplots(rows=6, cols=1, 
    #                     subplot_titles=[    'Deterministic Quick Sort with Sorted Input',
    #                                     'Deterministic Quick Sort with Randomized Input',
    #                                                'Random Quick Sort with Sorted Input',
    #                                            'Random Quick Sort with Randomized Input',
    #                                                       'Merge Sort with Sorted Input',
    #                                                   'Merge Sort with Randomized Input' ],
    #                     x_title="Samples",
    #                     y_title="Runtime (ms)")

    # x=list(cases.keys())

    # fig.add_trace(go.Scatter(x=x, y=det_qs_sorted_t, name='det_qs_sorted'),     row=1, col=1)
    # fig.add_trace(go.Scatter(x=x, y=det_qs_unsorted_t, name='det_qs_unsorted'), row=2, col=1)
    
    # fig.add_trace(go.Scatter(x=x, y=random_qs_sorted_t, name='random_qs_sorted'),     row=3, col=1)
    # fig.add_trace(go.Scatter(x=x, y=random_qs_unsorted_t, name='random_qs_unsorted'), row=4, col=1)
    
    # fig.add_trace(go.Scatter(x=x, y=merge_sort_sorted_t, name='merge_sort_sorted'),     row=5, col=1)
    # fig.add_trace(go.Scatter(x=x, y=merge_sort_unsorted_t, name='merge_sort_unsorted'), row=6, col=1)

    # h = 1000
    # fig.update_layout( height=h, width=h*1 
    #                   ,title_text="Algorithms - Input Size vs Execution Time",
    #                   template='plotly_dark',)
    # fig.show()

    # write_image(fig, 'Algo_Compare_Runtime_10k_Samples.pdf', format='pdf')

    t_total_duration = ti.time() - t_start
    print(f'\nTotal Runtime: {t_total_duration:0.03} s\n')
