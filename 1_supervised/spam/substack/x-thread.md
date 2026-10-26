# X thread

Post 1, then reply to yourself for 2 and 3. Attach `two_models_weights.png` on post 1 if you want the chart there.

---

Short recap of the simplest supervised models, using spam.

Labeled emails in. Train a few weights. New email in, classifier out.

Two versions, same job, same inference. (1) I pick 3 features; training only learns how much they matter. (2) One weight per word from the emails, start at 0. Counts aren't weights.

Both also have a bias: the guess when nothing in the email fires. Epoch 0 is always 0.50. After training the knobs actually mean something.

https://github.com/nirmal91/model-experiments

---

If you want one post instead of a thread, use only the block above as a single tweet (trim the last two paragraphs if it overflows).
