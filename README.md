# Linkping
Nvlink/RDMA bandwidth test

export PATH=/usr/local/cuda/bin:$PATH

## single

```text
./linkping single -k
```
```text
gpu  0   1   2 (GB/s
0    -   270 270 
1    270 -   270
2    270 270 -
```

## p2p

```text
./linkping p2p -s <CUDA:0> -d <CUDA:1> -s <size>
```

## server/client

```text
./linkping server -a <ip1> -s <size> -n <iteration>
```

```text
./linkping client -a <ip2> <ip1> -s <size> -n <iteration>
```
