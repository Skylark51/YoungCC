# Test isolation
Existing top-level regression tests are coordinator-owned.
For a feature, create narrowly scoped tests under tests/<module>/ rather
than rewriting another agent's tests. Reference real Gaussian fixture files
in place; never copy or modify source .out/.inp data.
The integration owner runs the complete test suite before merging.
