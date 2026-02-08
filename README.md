# kernels

kernels for practice

## Run

Build:

```
make build
```

Run:

```
make run
```

Optional sizes:

```
make run RUN_ARGS="512 512 256"
```

Override the default source:

```
make run SRC=path/to/file.cu
```

Or pass it as a positional arg:

```
make run path/to/file.cu
```
