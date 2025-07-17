# Linkping
Nvlink/RDMA bandwidth test

export PATH=/usr/local/cuda/bin:$PATH

## single

```text
./linkping single -k
```
```text
Linkping version 0.1.0
Linkping started in single mode.


rank 0, device 0, stream 0x7fcd70000c60, start 0x7fcd70000d20, end 0x7fcd70000fa0
rank 1, device 1, stream 0x7fcd68000c60, start 0x7fcd68000d20, end 0x7fcd68000fa0
rank 2, device 2, stream 0x7fcd6c000c60, start 0x7fcd6c000d20, end 0x7fcd6c000fa0
rank 3, device 3, stream 0x7fcd60000c60, start 0x7fcd60000d20, end 0x7fcd60000fa0
rank 4, device 4, stream 0x7fcd64000c60, start 0x7fcd64000d20, end 0x7fcd64000fa0
rank 5, device 5, stream 0x7fcd58000c60, start 0x7fcd58000d20, end 0x7fcd58000fa0
rank 6, device 6, stream 0x7fcd5c000c60, start 0x7fcd5c000d20, end 0x7fcd5c000fa0
rank 7, device 7, stream 0x7fcd50000c60, start 0x7fcd50000d20, end 0x7fcd50000fa0

P2P Bandwidth Topo Matrix (GB/s):
               0         1         2         3         4         5         6         7
   0 |        --    249.46    267.47    267.82    267.87    267.73    267.74    267.62
   1 |    267.65        --    236.62    267.68    267.81    267.74    267.68    267.39
   2 |    267.65    267.77        --    257.41    267.67    267.73    267.88    267.28
   3 |    267.67    267.84    267.51        --    229.58    267.74    267.86    267.39
   4 |    267.74    267.72    267.54    267.83        --    249.27    267.68    267.45
   5 |    267.73    267.72    267.57    267.64    267.66        --    229.53    267.22
   6 |    267.62    267.65    267.63    267.74    267.74    267.77        --    218.30
   7 |    219.66    267.67    267.57    267.86    267.66    267.73    267.75        --

Topo test finished.
```

## p2p

```text
./linkping p2p -s <CUDA:0> -d <CUDA:1> -s <size> -k <keep_runing> -n <iteration>

./linkping p2p
Linkping version 0.1.0
Linkping started in p2p mode.


----------------------------------------------------------------------------------------------------
srcRank: 0, dstRank: 1
Size: 10000000000
iters: 10
type: float
P2P can Access: Nvlink
----------------------------------------------------------------------------------------------------
#bytes              #iterations         #overhead(ms)       #bandwidth(GB/s)    #Total(GB)          
40000000000         1                   171.79              232.8411            40                  
40000000000         2                   171.80              232.8248            80                  
40000000000         3                   171.79              232.8357            120                 
40000000000         4                   171.80              232.8279            160                 
40000000000         5                   171.76              232.8869            200                 
40000000000         6                   171.76              232.8842            240   
```

## server/client

```text
./linkping server -a <ip1> -s <size> -n <iteration>
```

```text
./linkping client -a <ip2> <ip1> -s <size> -n <iteration>
```

*那么你为什么不用官方的nccl-test呢？*

**nvlink across Tray is about to be supported**