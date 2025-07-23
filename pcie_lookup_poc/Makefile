devel:
	python3 setup.py develop --user

bm:
	python3 test_embedding_lookup.py

bm_ncu:
	./ncu --set full -f -o nsys/custom_lookup.%p --nvtx --nvtx-include "custom_embedding_lookup/" --nvtx-include "custom_embedding_lookup_on_host_table/" --nvtx-include "custom_embedding_lookup_on_host_table_with_5sms/" python3 test_embedding_lookup.py
