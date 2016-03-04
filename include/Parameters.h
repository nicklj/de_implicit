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
#ifndef DE_IMPLICIT_PARAMETERS_H
#define DE_IMPLICIT_PARAMETERS_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/symmetric_tensor.h>

using namespace dealii;
using namespace std;

namespace Parameters
{
    struct Problem
    {
      string inp_file_name;
      double geo_x, geo_y, geo_z;
      double total_force, total_voltage;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    struct Geometry
    {
      unsigned int global_refinement;
      double       scale;
      double       p_p0;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };



    struct Materials
    {
      double mu;
      double kappa;
      double Jm;
      double epsilon;
      double rho;
      double pre_x, pre_y, pre_z;
      double En_x, En_y, En_z;
      double damp_c;
      double charge_density;
      
      double phi_scal;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };



    struct LinearSolver
    {
      std::string type_lin;
      double      tol_lin;
      double      max_iterations_lin;
      std::string preconditioner_type;
      double      preconditioner_relaxation;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    struct Time
    {
      double dt;
      double end_time;
      double dt_pre;
      double end_time_pre;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };



    struct AllParameters : 
      public Problem,
      public FESystem,
      public Geometry,
      public Materials,
      public LinearSolver,
      public NonlinearSolver,
      public Time

    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

}



// Initiate standard tensor
template <int dim>
class StandardTensors
{
    public:

    static const SymmetricTensor<2, dim> I;
    static const SymmetricTensor<4, dim> IxI;
    static const SymmetricTensor<4, dim> II;
    static const SymmetricTensor<4, dim> dev_P;
};

template <int dim>
const SymmetricTensor<2, dim>
StandardTensors<dim>::I = unit_symmetric_tensor<dim>();

template <int dim>
const SymmetricTensor<4, dim>
StandardTensors<dim>::IxI = outer_product(I, I);

template <int dim>
const SymmetricTensor<4, dim>
StandardTensors<dim>::II = identity_tensor<dim>();

template <int dim>
const SymmetricTensor<4, dim>
StandardTensors<dim>::dev_P = deviator_tensor<dim>();

#endif
