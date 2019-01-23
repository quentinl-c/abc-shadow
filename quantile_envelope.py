from multiprocessing import Pool, cpu_count


THREAD_NBR = cpu_count()
with Pool(processes=THREAD_NBR) as processes_pool:
    x =[(1,2), (3,4), (5,4), (7,9), (0,1)]
    processes_pool.starmap(t, x)
