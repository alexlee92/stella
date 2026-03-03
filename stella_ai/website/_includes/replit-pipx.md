To use stella with pipx on replit, you can run these commands in the replit shell:

```bash
pip install pipx
pipx run stella-chat ...normal stella args...
```

If you install stella with pipx on replit and try and run it as just `stella` it will crash with a missing `libstdc++.so.6` library.

