# Demo sites

## Important ARGs in Fingerprinting

The following two loops are the most important pieces of code for fingerprinting, and the number of loops is the key to controlling the time and effect of fingerprinting.

GPU fingerprinting code:

```c
float stall_function()
{
    float res = 0.01;
    for(int i = 1; i < GPU_FP_ARGS; i++)
    {
        res = sinh(res);
    }
    return res;
}     
```

CPU fingerprinting code:

```javascript
function stall_function_cpu(arg) {
    var array = new Uint32Array(arg);
    var start = performance.now();
    for (var k = 1; k <= CPU_FP_ARGS; k++) {
        crypto.getRandomValues(array);
    }
    var end = performance.now();
    return end - start;
}
```

## Settings of ARGs

Due to the differences in hardware characteristics between different devices and architectures, we set different parameters for different experiments, as follows:

| Settings | `GPU_FP_ARGS` | `CPU_FP_ARGS` |
| :------  | :-----------: | :-----------: |
| Windows | `0xfffff` | `20000` |
| Windows (Real World) | `0x3ffff` | `5000` |
| MacOS | `0xffff` | `5000` |
| Android | `0xffff` | `500` |
| Android (Real World) | `0xfff` | `500` |

## Note
The fp_for_gpu.html is modified on the basis of https://github.com/drawnapart/drawnapart.
