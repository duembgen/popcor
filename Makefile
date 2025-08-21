doc:
	sphinx-autobuild docs/source docs/build --watch popcor/

doctest:
	sphinx-build -b doctest docs/source docs/build/doctest
	
