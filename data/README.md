# Data

Datasets are not committed to git. Download them here.

## SIFT10K (development, 5 MB)

From the project root, in PowerShell:

```powershell
curl.exe -o data\siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf data\siftsmall.tar.gz -C data
```

This creates `data\siftsmall\` with `siftsmall_base.fvecs` (10,000 × 128), `siftsmall_query.fvecs` (100 × 128) and `siftsmall_groundtruth.ivecs` (100 × 100 true neighbour ids).

If the FTP download fails, get it from the dataset page: http://corpus-texmex.irisa.fr/ (ANN_SIFT10K).

## SIFT1M (final benchmarks, 161 MB — week 4, on Kaggle)

`ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` — too large to benchmark comfortably on the laptop; run it in a Kaggle notebook.
