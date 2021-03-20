import ray
import numpy as np

ray.init()



@ray.remote
def test_ray(i, j):
    x = []
    for _ in range(100):
        x.append(i+j)
        
    return x
futures = [test_ray.remote(x, x**2) for x in range(4)]

print(ray.get(futures))
