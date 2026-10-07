# Test fixtures

All files here are SYNTHETIC: the sentences were written for the tests and are not
taken from KazNERD or any other corpus. `kaznerd_tiny/` mimics the KazNERD file layout
(`IOB2_train.txt`, `IOB2_valid.txt`, `IOB2_test.txt`) and includes the edge cases the reader
must handle: a `0` label, repeated blank lines and trailing blank lines.
