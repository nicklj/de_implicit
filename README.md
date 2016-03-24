DE_implicit source code
=======================

This code is developed based on example step-44 of the deal.II FEM library, and
can be used (but not limited) for dielectric elastomer FEM simluation. 
It is the supplimentary code to the article 

"A 3D multi-field element for simulating dielectric elastomer (DE) electromechanical coupling behavior"

1. deal.II Requirements:
------------------------

The source code requires the deal.II 8.3.0 library or later for
compiling. For the program to work properly, deal.II should be 
configured with support for Trilinos, UMFPACK, MPI, and C++11. 
The user is responsible for correctly setup the deal.II environment 
for successfully compiling and running the program. 
The details on deal.II installation can be found in: 
https://dealii.org/

  
2. Installation
---------------

You can clone it from github

	git clone https://github.com/nicklj/de_implicit.git

The program can then be compiled by running

	cd de_implicit
	cmake .
	make

You can run the program by simply inputing

	./run


3. Simulation parameters
-----------------------

The program uses parameter files to set its runtime variables. 
This configuration file is located at: 
    ./parameters.prm


4. Licence Informations
-----------------------

Please see the file ./LICENSE for details

