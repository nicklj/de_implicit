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
#ifndef DE_IMPLICIT_DE_IMPLICIT_H
#define DE_IMPLICIT_DE_IMPLICIT_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/identity_matrix.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include "Parameters.h"

const double PI = 3.141592653589793238;

using namespace dealii;

// Define material constitutive model
template <int dim>
class Material_gent_DE
{
public:
    Material_gent_DE(const double mu, const double kappa, const double Jm, const double epsilon)
      :
      mu(mu),
      kappa(kappa),
      Jm(Jm),
      epsilon(epsilon),
      detF(1.0),
      b_bar(StandardTensors<dim>::I)
    {
      Assert(kappa > 0, ExcInternalError());
    }

    ~Material_gent_DE()
    {}


    void update_material_data(const Tensor<2, dim> &F, 
                              const Tensor<1, dim> &E_t,
                              const double p_tilde_in,
                              const double J_tilde_in)
    {
        double scal;

        detF = determinant(F); 
        F_inv = invert(F);
        this->E_t = E_t;

        // Calculate the distortion tensor
        scal = pow(detF, -1.0/3.0);
        F_dist = F * scal;
        b_bar = symmetrize(F_dist * transpose(F_dist));

        I1_bar = trace(b_bar);

        CK = 1.0 - (I1_bar - 3.0)/Jm;
        EG = mu/detF/CK;
        EK = kappa*(detF - 1.0/detF)/2.0;
        EkEk = E_t * E_t;

        p_tilde = p_tilde_in;
        J_tilde = J_tilde_in;
    }
       
    SymmetricTensor<2, dim> get_sigma_str() 
    {
        //Cauchy stress
        T_dev = EG*(b_bar - I1_bar/3.0*StandardTensors<dim>::I);
        T_vol = p_tilde*StandardTensors<dim>::I;

        return (T_dev + T_vol);
    }

    SymmetricTensor<2, dim> get_sigma_ele() 
    {
        for (unsigned int i=0;i<3;i++) {
            for (unsigned int j=0;j<=i;j++) {
                if (i == j)
                    T_Max[i][j] = epsilon*E_t[i]*E_t[j] - 0.5*epsilon*EkEk;
                else
                    T_Max[i][j] = epsilon*E_t[i]*E_t[j];
            }
        }
        return T_Max;
    }

    SymmetricTensor<2, dim> get_sigma() 
    {
        return get_sigma_str() + get_sigma_ele();
    }


    double get_J() const
    {
      return detF;
    }

    SymmetricTensor<4, dim> get_c_iso() const
    {
        double temp = 2.0*mu/detF/CK;
        double temp1 = temp * 1.0/CK/Jm;
        double temp2 = temp * (-I1_bar/3.0/CK/Jm-1.0/3.0);
        double temp3 = temp * (-I1_bar/3.0/CK/Jm-1.0/3.0);
        double temp4 = temp * (I1_bar*I1_bar/9.0/CK/Jm + I1_bar/9.0/CK);
        double temp5 = temp * I1_bar/6.0;
        SymmetricTensor<2, dim> I = StandardTensors<dim>::I;
        SymmetricTensor<4, dim> c_iso;
        for (unsigned int i=0;i<3;i++)
        for (unsigned int j=0;j<=i;j++)
        for (unsigned int k=0;k<3;k++)
        for (unsigned int l=0;l<=k;l++)
            c_iso[i][j][k][l] = temp1*b_bar[i][j]*b_bar[k][l]
                                + temp2*b_bar[i][j]*I[k][l]
                                + temp3*I[i][j]*b_bar[k][l]
                                + temp4*I[i][j]*I[k][l]
                                + temp5*(I[i][k]*I[j][l] + I[i][l]*I[j][k]);

        return c_iso;
    }

    SymmetricTensor<4, dim> get_c_vol() const
    {
        return p_tilde * (StandardTensors<dim>::IxI - 2.0*StandardTensors<dim>::II);
    }

    SymmetricTensor<4, dim> get_c_ele() const
    {
        SymmetricTensor<2, dim> I = StandardTensors<dim>::I;
        SymmetricTensor<4, dim> c_ele;
        for (unsigned int i=0;i<3;i++)
        for (unsigned int j=0;j<=i;j++)
        for (unsigned int k=0;k<3;k++)
        for (unsigned int l=0;l<=k;l++)
            c_ele[i][j][k][l] = 0.5*epsilon*(I[i][k]*I[j][l] + I[i][l]*I[j][k] - I[i][j]*I[k][l])*EkEk
                                + epsilon*(I[k][l]*E_t[i]*E_t[j] + I[i][j]*E_t[k]*E_t[l]
                                          -I[i][l]*E_t[j]*E_t[k] - I[i][k]*E_t[j]*E_t[l]
                                          -I[j][l]*E_t[i]*E_t[k] - I[j][k]*E_t[i]*E_t[l]);
        return c_ele;
    }

    SymmetricTensor<4, dim> get_c() const
    {
      return get_c_vol() + get_c_iso() + get_c_ele();
    }

    double get_dPsi_vol_dJ() const
    {
        return (kappa/2.0)*(J_tilde - 1.0/J_tilde);
    }

    double get_d2Psi_vol_dJ2() const
    {
        return (kappa/2.0)*(1.0 + 1.0/(J_tilde*J_tilde));
    }

    double get_p_tilde() const
    {
      return p_tilde;
    }

    double get_J_tilde() const
    {
      return J_tilde;
    }

protected:
    const double mu;
    const double kappa;
    const double Jm;
    const double epsilon;

    double detF;
    double p_tilde, J_tilde;
    Tensor<2, dim> F_dist, F_inv;
    Tensor<1, dim> E_t;
    SymmetricTensor<2, dim> b_bar, C, T, T_dev, T_vol, T_Max;
    double I1_bar, CK, EG, EK, EkEk;


};

/////////////////////////////////////////////////////////////////////////////////
// Define point history class
template <int dim>
class PointHistory
{
    public:
    PointHistory()
      :
      material(NULL),
      F(StandardTensors<dim>::I),
      F_inv(StandardTensors<dim>::I),
      sigma(SymmetricTensor<2, dim>()),
      d2Psi_vol_dJ2(0.0),
      dPsi_vol_dJ(0.0),
      c(SymmetricTensor<4,dim>())
    {}

    virtual ~PointHistory()
    {
      delete material;
      material = NULL;
    }

    void setup_lqp (const Parameters::AllParameters &parameters)
    {
      material = new Material_gent_DE<dim>(parameters.mu, parameters.kappa, parameters.Jm, parameters.epsilon);
      update_values(Tensor<2, dim>(), //F
                    StandardTensors<dim>::I, //Prestretch
                    Tensor<1, dim>(), //E_t
                    0.0,              //pressure
                    1.0);             //dilation
    }

    // update all the related values that can be used for assembling stiffness matrix
    void update_values (const Tensor<2, dim> &Grad_u, 
                        const Tensor<2, dim> &PRES,
                        const Tensor<1, dim> &Grad_phi,
                        const double p_tilde,
                        const double J_tilde)
    {
        F = Tensor<2, dim>(StandardTensors<dim>::I) + Grad_u;
        FP = update_pre_stretch(F, PRES); // Update pre-stretch
        F_inv = invert(F);
        E_t = -transpose(F_inv)*Grad_phi; 
        material->update_material_data(FP, E_t, p_tilde, J_tilde);
        sigma = material->get_sigma();
        c = material->get_c();
        dPsi_vol_dJ = material->get_dPsi_vol_dJ();
        d2Psi_vol_dJ2 = material->get_d2Psi_vol_dJ2();
    }

    // Interface for getting values
    double get_J() const
    {
      return material->get_J();
    }

    double get_p_tilde() const
    {
      return material->get_p_tilde();
    }

    double get_J_tilde() const
    {
      return material->get_J_tilde();
    }

    const Tensor<2, dim> &get_F() const
    {
      return F;
    }

    const Tensor<1, dim> &get_E_t() const
    {
      return E_t;
    }

    const Tensor<2, dim> &get_F_inv() const
    {
      return F_inv;
    }

    const SymmetricTensor<2, dim> &get_sigma() const
    {
      return sigma;
    }

    const SymmetricTensor<4, dim> &get_c() const
    {
      return c;
    }
    
    double get_dPsi_vol_dJ() const
    {
        return dPsi_vol_dJ;
    }

    double get_d2Psi_vol_dJ2() const
    {
        return d2Psi_vol_dJ2;
    }

private:
    Material_gent_DE<dim> *material;
    Tensor<1, dim> E_t;
    Tensor<2, dim> F, F_inv, FP;
    SymmetricTensor<2, dim> sigma;
    double d2Psi_vol_dJ2, dPsi_vol_dJ;
    SymmetricTensor<4, dim> c;

    Tensor<2, dim> update_pre_stretch(const Tensor<2, dim> &F, 
                                      const Tensor<2, dim> &PRES)
    {
        // Apply prestretch
        return F * PRES;
    }
};

////////////////////////////////////////////////////////////
class Time
{
    public:
    Time (const double time_end, const double dt, const bool ispre = false)
      :
      timestep(0),
      time_current(0.0),
      time_end(time_end),
      dt(dt),
      ispre(ispre)
    {}

    virtual ~Time()
    {}

    double current() const
    {
      return time_current;
    }
    double end() const
    {
      return time_end;
    }
    double get_dt() const
    {
      return dt;
    }
    unsigned int get_timestep() const
    {
      return timestep;
    }
    unsigned int get_total_timestep() const
    {
      return round(time_end/dt);
    }
    void increment()
    {
      time_current += dt;
      ++timestep;
    }
    void rollback()
    {
      time_current -= dt;
      --timestep;
      dt = dt/2.0;
    }

    void reset()
    {
      time_current = 0.0;
    }

    bool is_pre()
    {
      return ispre;
    }

    private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    double dt;
    const bool ispre;
};

/////////////////////////////////////////////////////////////////////////////////
    
struct Geometry
{
    double x,y,z,scale;
    
    Geometry (double x, double y, double z, double scale)
    :
        x(x*scale),y(y*scale),z(z*scale),scale(scale)
    {}
};

//Decalration of Solid class
template <int dim>
class Solid
{
    public:
    Solid(const std::string &input_file);
    Solid() {}

    virtual
    ~Solid();

    void
    run();

    private:

    /* Data structures used for parallelizing with TBB*/
    struct PerTaskData_K;
    struct ScratchData_K;

    struct PerTaskData_RHS;
    struct ScratchData_RHS;

    struct PerTaskData_SC;
    struct ScratchData_SC;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    // Set up the finite element system to be solved:
    void system_setup();
    void determine_component_extractors();

    // Tangent K matrix
    void assemble_system_tangent();
    void assemble_system_tangent_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData_K &scratch, PerTaskData_K &data);
    void copy_local_to_global_K(const PerTaskData_K &data);

    // RHS matrix
    void assemble_system_rhs();
    void assemble_system_rhs_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 ScratchData_RHS &scratch, PerTaskData_RHS &data);
    void copy_local_to_global_rhs(const PerTaskData_RHS &data);

    // Static condensed matrix
    void assemble_sc();
    void assemble_sc_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                             ScratchData_SC &scratch,
                             PerTaskData_SC &data);
    void copy_local_to_global_sc(const PerTaskData_SC &data);

    // Apply Dirichlet boundary conditions on the displacement field
    void make_constraints(const int &it_nr);

    // Create and update the quadrature points.
    void setup_qph();
    void update_qph_incremental(const BlockVector<double> &solution_delta);
    void update_qph_incremental_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                    ScratchData_UQPH &scratch, PerTaskData_UQPH &data);
    void copy_local_to_global_UQPH(const PerTaskData_UQPH &/*data*/)
        {}

    // Solve for the displacement using a Newton-Raphson method.
    void solve_nonlinear_timestep(BlockVector<double> &solution_delta,
                                  Time &time);
    std::pair<unsigned int, double> solve_linear_system(BlockVector<double> &newton_update);

    // Solution retrieval as well as post-processing and writing data to file:
    BlockVector<double> get_total_solution(const BlockVector<double> &solution_delta) const;

    void make_grid();

    void output_results(Time &time) const;
    void output_eigenvector() const;

    void assemble_mass_matrix();
    void solve_eig();

    /////////////////////////////////////////////////////////////////////////////////

    Parameters::AllParameters        parameters;
    Triangulation<dim>               triangulation;

    Time                             time;
    Time                             time_pre;
    TimerOutput                      timer;

    std::vector<PointHistory<dim>>   quadrature_point_history;

    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler_ref;
    DoFHandler<dim>                  dof_handler_xi;
    const unsigned int               dofs_per_cell;

    const FEValuesExtractors::Vector  u_fe;
    const FEValuesExtractors::Scalar  phi_fe;
    const FEValuesExtractors::Scalar  p_fe;
    const FEValuesExtractors::Scalar  J_fe;

    static const unsigned int        n_blocks = 3;
    static const unsigned int        n_components = dim + 3;
    static const unsigned int        first_u_component = 0;
    static const unsigned int        phi_component = dim;
    static const unsigned int        p_component = dim + 1;
    static const unsigned int        J_component = dim + 2;

    enum
    {
        u_dof = 0,
        phi_dof = 1,
        p_dof = 2,
        J_dof = 3,
    };

    enum
    {
        xi_blk = 0,
        p_blk = 1,
        J_blk = 2,
    };

    std::vector<types::global_dof_index>    dofs_per_block;

    std::vector<types::global_dof_index>    element_indices_u;
    std::vector<types::global_dof_index>    element_indices_phi;
    std::vector<types::global_dof_index>    element_indices_xi;
    std::vector<types::global_dof_index>    element_indices_p;
    std::vector<types::global_dof_index>    element_indices_J;

    std::vector<types::global_dof_index>    total_indices_u;
    std::vector<types::global_dof_index>    total_indices_phi;

    const QGauss<dim>                qf_cell;
    const QGauss<dim-1>              qf_face;

    const unsigned int               n_q_points;
    const unsigned int               n_q_points_f;

    ConstraintMatrix                 constraints;
    BlockSparsityPattern             sparsity_pattern;
    BlockSparseMatrix<double>        tangent_matrix;
    BlockSparseMatrix<double>        mass_matrix;
    BlockVector<double>              system_rhs;
    BlockVector<double>              solution_n;
    BlockVector<double>              solution_n_loading;

    std::vector<double> probe_node_stretch;
    std::vector<double> probe_node_potential;
    types::global_dof_index  probe_vertex_x;
    types::global_dof_index  probe_vertex_z;
    types::global_dof_index  probe_vertex_x_2;
    types::global_dof_index  probe_vertex_z_2;
    types::global_dof_index  probe_vertex_phi;


    struct Errors
    {
        Errors()
          :
          norm(1.0), u(1.0), phi(1.0), p(1.0), J(1.0)
        {}

        void reset()
        {
            norm = 1.0;
            u = 1.0;
            phi = 1.0;
            p = 1.0;
            J = 1.0;
        }
        
        void normalise(const Errors &rhs)
        {
            if (rhs.norm != 0.0)
                norm /= rhs.norm;
            if (rhs.u != 0.0)
                u /= rhs.u;
            if (rhs.phi != 0.0)
                phi /= rhs.phi;
            if (rhs.p != 0.0)
                p /= rhs.p;
            if (rhs.J != 0.0)
                J /= rhs.J;
        }

        double norm, u, phi, p, J;
    };

    Errors error_residual, error_residual_0, error_residual_norm, 
           error_update,   error_update_0,   error_update_norm;
    
    void get_error_residual(Errors &error_residual);
    void get_error_update(const BlockVector<double> &newton_update,
                          Errors &error_update);
    double get_l2_norm(const std::vector<double> &vec);

    void print_msg_header();
    void print_msg_footer();

    Tensor<2, dim> PRES;

    Geometry geo;

    std::set<types::global_dof_index> leftmid_zdof;
};


#endif
