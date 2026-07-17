# Negative controls


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_37d72c9efedd", "created_at": "2026-07-17T04:04:04+00:00", "title": "Three fail-closed controls are rejected: omit reaching probability (L1 0.157),…"}
-->
Three fail-closed controls are rejected: omit reaching probability (L1 0.157), omit partition factors (0.00933), and apply beta=2 composition to a beta=1 target (0.0702). The 512-setting reaching ablation has median L1 0.115.


---
<!-- trackio-cell
{"type": "code", "id": "cell_73e7c8e83c9a", "created_at": "2026-07-17T04:04:10+00:00", "title": "Seventeen tests", "command": ["pytest", "-q"], "exit_code": 0, "duration_s": 5.346}
-->
````bash
$ pytest -q
````

exit 0 · 5.3s


````output
.................                                                        [100%]
17 passed in 5.10s

````
