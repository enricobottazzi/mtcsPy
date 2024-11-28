from mtcspy.mtcs import add
import timeit

def benchmark_add():

    start = timeit.default_timer()
    for _ in range(1000000):
        add(1, 2)
    end = timeit.default_timer()

    print(f"Time: {end - start} seconds")
    
if __name__ == "__main__":
    benchmark_add()