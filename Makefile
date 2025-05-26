doc:
	sphinx-autobuild docs/source docs/build

doctest:
	sphinx-build -b doctest docs/source docs/build/doctest
	
