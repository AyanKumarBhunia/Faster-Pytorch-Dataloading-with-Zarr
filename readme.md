
# ⚡️ Zarr Tutorial — Fast, Chunked, Lossless Array Storage for ML and Vision

Zarr is a **chunked, compressed, N-dimensional array format** designed for high-performance numerical data storage.  
It’s ideal for deep learning datasets involving large video or image stacks (e.g., RGB frames, optical flow, depth maps), where random crop access and throughput matter more than human readability.

---

## 🧠 What Is Zarr?

Zarr stores NumPy-like arrays directly on disk or in cloud object stores, in **regularly chunked blocks**, each optionally **compressed**.

Each array is accompanied by lightweight **metadata** (`.zarray` JSON), describing shape, dtype, compressor, and chunk layout.

### Example Layout
```
my_dataset.zarr/
├── .zgroup
├── gt/
│   ├── 0.0.0
│   ├── 0.0.1
│   ├── ...
│   └── .zarray
└── lr/
    ├── 720x1280/
    │   ├── rgb/
    │   │   ├── 0.0.0
    │   │   ├── ...
    │   │   └── .zarray
    ├── flow/
    └── depth/
```

Each file (like `0.0.0`) is one compressed chunk of numeric data.

---

## 🚀 Why Zarr Is Fast

| Reason | Explanation |
|--------|--------------|
| 🧱 **Chunked storage** | Reads only the subset of chunks overlapping your crop (e.g., 960×960 region). PNG/TIFF must decode entire images. |
| ⚙️ **Native binary format** | Stores `uint8`, `float16`, `float32`, etc., directly — no image decode or color conversion. |
| 💨 **Parallel I/O and compression** | Uses [Blosc](https://blosc.org) (LZ4/Zstd), which decompresses multiple chunks in parallel. |
| 📖 **Random access** | Load arbitrary crops instantly with array slicing. |
| 🚫 **No filesystem overhead** | One directory per array, not millions of small files. |
| ☁️ **Cloud-ready** | Works seamlessly with S3, GCS, or local NVMe. |

---

## 🧩 Installing

```bash
pip install zarr numcodecs
```

Optional:
```bash
pip install nvidia-ml-py3  # if you want to monitor GPU usage later
```

---

## 🧪 Creating a Zarr Dataset

Below we create a dataset containing 100 RGB frames (`uint8`) at 4K resolution, chunked for 960×960 crops.

```python
import numpy as np
import zarr
import numcodecs

# Configure compressor (Zstandard is a good default)
compressor = numcodecs.Blosc(
    cname='zstd',
    clevel=5,
    shuffle=numcodecs.Blosc.BITSHUFFLE
)

# Create the root group
root = zarr.open_group('demo.zarr', mode='w')

# Create an array: 100 frames, 3 channels, 2160×3840
gt = root.create_dataset(
    'gt',
    shape=(100, 3, 2160, 3840),
    chunks=(1, 3, 960, 960),
    dtype='uint8',
    compressor=compressor
)

# Write some random data
gt[:] = np.random.randint(0, 255, gt.shape, np.uint8)
print("Zarr dataset written to demo.zarr/")
```

✅ Each chunk (`960×960×3`) is stored and compressed independently, so reading one crop later is instantaneous.

---

## 🔍 Reading Crops from Zarr

```python
import zarr
import numpy as np

z = zarr.open_group('demo.zarr', mode='r')
print("Shape:", z['gt'].shape)

# Read a 960×960 crop at random position
t = 0
top, left = 500, 1000
patch = z['gt'][t, :, top:top+960, left:left+960]  # CHW order
print("Crop shape:", patch.shape)
```

This reads and decompresses **only the few chunks** that overlap your crop.

---

## ⚖️ Zarr vs. PNG / TIFF

| Feature | PNG | TIFF | Zarr |
|----------|------|------|------|
| Type | Image (2D) | Image (2D/3D) | N-dim array |
| Compression | Deflate | LZW, ZSTD | Blosc (LZ4/ZSTD) |
| Random crop access | ❌ Full decode | ⚠️ Whole tiles | ✅ Chunk-level reads |
| I/O parallelism | ❌ | ⚠️ Limited | ✅ Native multithreaded |
| Data type support | 8-bit | 8/16/32-bit | any NumPy dtype |
| Lossless? | Yes | Yes | Yes (default) |
| Performance | Slow | Medium | Very fast |
| Ideal for | Archival | Imaging | ML training & analysis |

Zarr is not “visually lossless” like PNG — it’s **numerically lossless** for your stored dtype.  
If you write `float32`, you get back *exactly the same float32 values* after decompression.

---

## 💾 Example: Multiresolution Dataset

You can store multiple resolutions in one hierarchy:

```python
root = zarr.open_group('multi_res.zarr', mode='w')

for res, shape in [('720p', (100,3,720,1280)),
                   ('1080p', (100,3,1080,1920)),
                   ('2160p', (100,3,2160,3840))]:
    g = root.require_group(f'lr/{res}')
    g.create_dataset('rgb', shape=shape, chunks=(1,3,360,640),
                     dtype='uint8', compressor=compressor)
    g.create_dataset('flow', shape=(100,2,shape[2],shape[3]),
                     chunks=(1,2,360,640), dtype='float16',
                     compressor=compressor)
```

Now you can read crops at any resolution level with:
```python
patch_1080 = root['lr/1080p/rgb'][5, :, 200:1160, 300:1500]
```

---

## ⚙️ Typical Compressor Settings

| Compressor | Use case | Notes |
|-------------|-----------|-------|
| **LZ4** | Maximum speed, large files | Extremely fast, modest compression |
| **ZSTD (zstandard)** | Balanced speed/size | Best all-rounder |
| **BITSHUFFLE** | Good for float data | Reorders bytes for better compressibility |
| **float16 dtype** | Halves size of flow/depth data | Still precise enough for most ML uses |

All are **lossless** unless you manually quantize (e.g., convert float32→float16).

---

## 🔄 Random Crop Access Example

```python
import zarr, torch, random
z = zarr.open_group('multi_res.zarr', mode='r')

def random_crop(res='2160p', size=960):
    t = random.randint(0, z['gt'].shape[0] - 1)
    H, W = z['gt'].shape[2:]
    top  = random.randint(0, H - size)
    left = random.randint(0, W - size)
    crop = z[f'lr/{res}/rgb'][t, :, top:top+size, left:left+size]
    return torch.from_numpy(crop)

patch = random_crop()
print(patch.shape)
```

---

## 🧩 Best Practices for ML Datasets

| Tip | Why it matters |
|-----|----------------|
| **Chunk size ≈ crop size** | Keeps each read within 1–4 chunks (fastest I/O). |
| **Use Blosc+Zstd, clevel=5** | Great compression/speed trade-off. |
| **Store uint8 for RGB, float16 for flow/depth** | Cuts size ~3× with no loss. |
| **Avoid millions of tiny files** | Zarr stores all frames/channels neatly under one folder. |
| **Use persistent_workers in DataLoader** | Prevents repeated open/close of Zarr store. |

---

## 🧭 Quick Performance Test

```python
import time, random
import numpy as np, zarr

z = zarr.open_group('demo.zarr', mode='r')
N = 100
t0 = time.time()
for _ in range(N):
    top, left = random.randint(0, 1200), random.randint(0, 2000)
    _ = z['gt'][0, :, top:top+960, left:left+960]
print(f"{N} crops loaded in {time.time()-t0:.2f}s → {(N/(time.time()-t0)):.1f} crops/s")
```

On an NVMe SSD, Zarr can easily exceed **hundreds of 960×960 crops per second** — orders of magnitude faster than reading and decoding PNG/TIFF.

---

## ✅ Summary

| Feature | Zarr |
|----------|------|
| **Compression** | Lossless (Blosc LZ4/ZSTD) |
| **Access pattern** | Random, chunk-wise |
| **Speed** | Extremely fast (no decoding) |
| **I/O efficiency** | Excellent on NVMe and cloud |
| **Best for** | Large ML datasets, 3D or temporal data, random crops |

---

## 📚 References

- [Zarr specification](https://zarr.readthedocs.io)
- [Numcodecs compressors](https://numcodecs.readthedocs.io)
- [Blosc documentation](https://blosc.org/pages/blosc-in-depth/)

---

### TL;DR
**Zarr = NumPy + compression + chunking.**  
It gives you **lossless, high-throughput, random access** to large datasets — perfect for machine learning training and analysis workloads.
