/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2016 by Liu Jun
 *
 * This file is part of the DE_implicit program, and
 * is developed based on example step-44 of the deal.II FEM library
 * For more information, please refer: hpttp://www.dealii.org
 *
 * This program is developed for dielectric elastomer FEM simluation, 
 * and is free software; you can use it, redistribute it, and/or 
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either version
 * 2.1 of the License, or (at your option) any later version.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Liu Jun
 * Institute of High Performance Computing, A*STAR, Singapore, 138632
 * E-mail: liuj@ihpc.a-star.edu.sg
 *
 * ---------------------------------------------------------------------
 */
#include "Parameters.h"
#include <iostream>
#include <fstream>

using namespace dealii;
using namespace std;


namespace Parameters
{
    void Problem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Problem Description");
      {
        prm.declare_entry("Input file", "",
                          Patterns::FileName(),
                          "Input mesh name in ucd");

        prm.declare_entry("Geo x", "0.0",
                          Patterns::Double(),
                          "Geometry x");

        prm.declare_entry("Geo y", "0.0",
                          Patterns::Double(),
                          "Geometry y");

        prm.declare_entry("Geo z", "0.0",
                          Patterns::Double(),
                          "Geometry z");

        prm.declare_entry("Total force", "0.0",
                          Patterns::Double(),
                          "Total external pressure applied for pre-stretch");

        prm.declare_entry("Total voltage", "0.0",
                          Patterns::Double(),
                          "Total voltage");
      }
      prm.leave_subsection();
    }

    void Problem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Problem Description");
      {
        inp_file_name = prm.get("Input file");
        geo_x = prm.get_double("Geo x");
        geo_y = prm.get_double("Geo y");
        geo_z = prm.get_double("Geo z");
        total_force = prm.get_double("Total force");
        total_voltage = prm.get_double("Total voltage");
      }
      prm.leave_subsection();
    }

    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree", "1",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Quadrature order", "2",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }



    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Global refinement", "4",
                          Patterns::Integer(0),
                          "Global refinement level");

        prm.declare_entry("Grid scale", "1e-3",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");

        //prm.declare_entry("Pressure ratio p/p0", "100",
                          //Patterns::Selection("20|40|60|80|100"),
                          //"Ratio of applied pressure to reference pressure");
      }
      prm.leave_subsection();
    }

    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        global_refinement = prm.get_integer("Global refinement");
        scale = prm.get_double("Grid scale");
        //p_p0 = prm.get_double("Pressure ratio p/p0");
      }
      prm.leave_subsection();
    }



    void Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("Shear modulus", "4500",
                          Patterns::Double(),
                          "Shear modulus");

        prm.declare_entry("Bulk modulus", "4.5e8",
                          Patterns::Double(),
                          "Bulk modulus");

        prm.declare_entry("Jm", "120",
                          Patterns::Double(),
                          "Jm");

        prm.declare_entry("Epsilon", "3.98e-11",
                          Patterns::Double(),
                          "Epsilon");

        prm.declare_entry("Density", "1000.0",
                          Patterns::Double(),
                          "Density");

        prm.declare_entry("Pre-stretch x", "1",
                          Patterns::Double(),
                          "Pre-stretch x");

        prm.declare_entry("Pre-stretch y", "1",
                          Patterns::Double(),
                          "Pre-stretch y");

        prm.declare_entry("Pre-stretch z", "1",
                          Patterns::Double(),
                          "Pre-stretch z");

        prm.declare_entry("En_x", "0.0",
                          Patterns::Double(),
                          "En_x");

        prm.declare_entry("En_y", "0.0",
                          Patterns::Double(),
                          "En_y");

        prm.declare_entry("En_z", "0.0",
                          Patterns::Double(),
                          "En_z");

        prm.declare_entry("Damping parameter", "0.0",
                          Patterns::Double(),
                          "Damping");

        prm.declare_entry("Surface charge density", "0.0",
                          Patterns::Double(),
                          "Surface charge density");
      }
      prm.leave_subsection();
    }

    void Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        mu = prm.get_double("Shear modulus");
        kappa = prm.get_double("Bulk modulus");
        Jm = prm.get_double("Jm");
        epsilon = prm.get_double("Epsilon");
        rho = prm.get_double("Density");
        pre_x = prm.get_double("Pre-stretch x");
        pre_y = prm.get_double("Pre-stretch y");
        pre_z = prm.get_double("Pre-stretch z");
        En_x = prm.get_double("En_x");
        En_y = prm.get_double("En_y");
        En_z = prm.get_double("En_z");
        damp_c = prm.get_double("Damping parameter");
        charge_density = prm.get_double("Surface charge density");

        phi_scal = sqrt(epsilon/mu);
      }
      prm.leave_subsection();
    }



    void LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type", "Direct",
                          Patterns::Selection("GMRES|Direct"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Residual", "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry("Max iteration multiplier", "1",
                          Patterns::Double(0.0),
                          "Linear solver iterations (multiples of the system matrix size)");

        prm.declare_entry("Preconditioner type", "ssor",
                          Patterns::Selection("jacobi|ssor"),
                          "Type of preconditioner");

        prm.declare_entry("Preconditioner relaxation", "0.65",
                          Patterns::Double(0.0),
                          "Preconditioner relaxation value");
      }
      prm.leave_subsection();
    }

    void LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        type_lin = prm.get("Solver type");
        tol_lin = prm.get_double("Residual");
        max_iterations_lin = prm.get_double("Max iteration multiplier");
        preconditioner_type = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
      }
      prm.leave_subsection();
    }



    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson", "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force", "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement", "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f = prm.get_double("Tolerance force");
        tol_u = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
    }



    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("Pre End time", "1",
                          Patterns::Double(),
                          "End time for pre-stretch");

        prm.declare_entry("Pre Time step size", "0.01",
                          Patterns::Double(),
                          "Time step size for pre-stretch");

        prm.declare_entry("End time", "1",
                          Patterns::Double(),
                          "End time");

        prm.declare_entry("Time step size", "0.01",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }

    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time_pre = prm.get_double("Pre End time");
        dt_pre = prm.get_double("Pre Time step size");
        end_time = prm.get_double("End time");
        dt = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }



    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.read_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Problem::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Problem::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
    }

}
