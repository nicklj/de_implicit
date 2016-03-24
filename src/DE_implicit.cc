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
#include "DE_implicit.h"

using namespace dealii;

namespace deprog
{
//Construction for the solid class
template <int dim>
Solid<dim>::Solid(const std::string &input_file)
    :
    parameters(input_file),
    triangulation(),
    time(parameters.end_time, parameters.dt),
    time_pre(parameters.end_time_pre, parameters.dt_pre, true),
    timer(std::cout,
          TimerOutput::summary,
          TimerOutput::wall_times),
    degree(parameters.poly_degree),
    fe(FE_Q<dim>(parameters.poly_degree),               dim,
       FE_Q<dim>(parameters.poly_degree),               1, 
       FE_DGPMonomial<dim>(parameters.poly_degree - 1), 1, 
       FE_DGPMonomial<dim>(parameters.poly_degree - 1), 1), 
    dof_handler_ref(triangulation),
    dofs_per_cell(fe.dofs_per_cell),
    u_fe(first_u_component),
    phi_fe(phi_component),
    p_fe(p_component),
    J_fe(J_component),
    dofs_per_block(n_blocks),
    qf_cell(parameters.quad_order),
    qf_face(1),
    n_q_points(qf_cell.size()),
    n_q_points_f(qf_face.size()),
    probe_node_stretch(time.get_total_timestep()+1),
    probe_node_potential(time.get_total_timestep()+1),
    PRES(StandardTensors<dim>::I),
    geo(parameters.geo_x, parameters.geo_y, parameters.geo_z, parameters.scale)
{
    determine_component_extractors();
}

//Destruction of Solid class
template <int dim>
Solid<dim>::~Solid()
{
    dof_handler_ref.clear();
}

template <int dim>
void Solid<dim>::determine_component_extractors()
{
    element_indices_u.clear();
    element_indices_phi.clear();
    element_indices_p.clear();
    element_indices_xi.clear();
    element_indices_J.clear();

    for (unsigned int k = 0; k < fe.dofs_per_cell; k++) {
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        if (k_group == u_dof) {
            element_indices_u.push_back(k);
            element_indices_xi.push_back(k);
        }
        else if (k_group == phi_dof) {
            element_indices_phi.push_back(k);
            element_indices_xi.push_back(k);
        }
        else if (k_group == p_dof)
            element_indices_p.push_back(k);
        else if (k_group == J_dof)
            element_indices_J.push_back(k);
        else
            Assert(k_group <= phi_dof, ExcInternalError());
    }

}

/**********************************************************************************/
// Data structure for quadrature point history storage
template <int dim>
struct Solid<dim>::PerTaskData_UQPH
{
    void reset()
    {}
};

template <int dim>
struct Solid<dim>::ScratchData_UQPH
{
    const BlockVector<double>   &solution_total;
    std::vector<Tensor<2, dim> > solution_Grads_u_total;
    std::vector<Tensor<1, dim> > solution_Grad_phi_total;
    std::vector<double> solution_values_p_total;
    std::vector<double> solution_values_J_total;

    FEValues<dim>                fe_values_ref;
    ScratchData_UQPH(const FiniteElement<dim> &fe_cell,
                     const QGauss<dim> &qf_cell,
                     const UpdateFlags uf_cell,
                     const BlockVector<double> &solution_total)
      :
      solution_total(solution_total),
      solution_Grads_u_total(qf_cell.size()),
      solution_Grad_phi_total(qf_cell.size()),
      solution_values_p_total(qf_cell.size()),
      solution_values_J_total(qf_cell.size()),
      fe_values_ref(fe_cell, qf_cell, uf_cell)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      :
      solution_total(rhs.solution_total),
      solution_Grads_u_total(rhs.solution_Grads_u_total),
      solution_Grad_phi_total(rhs.solution_Grad_phi_total),
      solution_values_p_total(rhs.solution_values_p_total),
      solution_values_J_total(rhs.solution_values_J_total),
      fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags())
    {}

    void reset()
    {
      const unsigned int n_q_points = solution_Grads_u_total.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          solution_Grads_u_total[q] = 0.0;
          solution_Grad_phi_total[q] = 0.0;
          solution_values_p_total[q] = 0.0;
          solution_values_J_total[q] = 0.0;
        }
    }
};

//Initialize the quadrature point history storage
template <int dim>
void Solid<dim>::setup_qph()
{
    std::cout << " Setting up quadrature point data..." << std::endl;
    {
        triangulation.clear_user_data();
        {
            std::vector<PointHistory<dim> > tmp;
            tmp.swap(quadrature_point_history);
        }
        quadrature_point_history.resize(triangulation.n_active_cells() * n_q_points);
        unsigned int history_index = 0;
        for (typename Triangulation<dim>::active_cell_iterator cell =
        triangulation.begin_active(); cell != triangulation.end(); ++cell)
        {
            cell->set_user_pointer(&quadrature_point_history[history_index]);
            history_index += n_q_points;
        }
        Assert(history_index == quadrature_point_history.size(),
        ExcInternalError());
    }

    for (typename Triangulation<dim>::active_cell_iterator cell =
                triangulation.begin_active(); cell != triangulation.end(); ++cell)
    {
        PointHistory<dim> *lqph = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
        Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
        Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            lqph[q_point].setup_lqp(parameters);
    }
}

//Update the quadrature point history storage
template <int dim>
void Solid<dim>::update_qph_incremental(const BlockVector<double> &solution_delta)
{
    timer.enter_subsection("Update QPH data");
    std::cout << " UQPH " << std::flush;

    const BlockVector<double> solution_total(get_total_solution(solution_delta));

    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH per_task_data_UQPH;
    ScratchData_UQPH scratch_data_UQPH(fe, qf_cell, uf_UQPH, solution_total);

    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::update_qph_incremental_one_cell,
                    &Solid::copy_local_to_global_UQPH,
                    scratch_data_UQPH,
                    per_task_data_UQPH);

    timer.leave_subsection();
}

//Update the quadrature point history in one element
template <int dim>
void
Solid<dim>::update_qph_incremental_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                            ScratchData_UQPH &scratch, 
                                            PerTaskData_UQPH & /*data*/)
{
    PointHistory<dim> *lqph = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
    Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
    Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());
    Assert(scratch.solution_Grads_u_total.size() == n_q_points, ExcInternalError());
    Assert(scratch.solution_Grad_phi_total.size() == n_q_points, ExcInternalError());
    Assert(scratch.solution_values_p_total.size() == n_q_points, ExcInternalError());
    Assert(scratch.solution_values_J_total.size() == n_q_points, ExcInternalError());

    scratch.reset();

    scratch.fe_values_ref.reinit(cell);
    scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
                                                       scratch.solution_Grads_u_total);
    scratch.fe_values_ref[phi_fe].get_function_gradients(scratch.solution_total,
                                                         scratch.solution_Grad_phi_total);
    scratch.fe_values_ref[p_fe].get_function_values(scratch.solution_total,
                                                         scratch.solution_values_p_total);
    scratch.fe_values_ref[J_fe].get_function_values(scratch.solution_total,
                                                         scratch.solution_values_J_total);
    // Update quadrature point history according to the u gradient
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        lqph[q_point].update_values(scratch.solution_Grads_u_total[q_point],
                                    PRES,
                                    scratch.solution_Grad_phi_total[q_point],
                                    scratch.solution_values_p_total[q_point],
                                    scratch.solution_values_J_total[q_point]);
}

/**********************************************************************************/

// Data structure for assembing K for parallizing with TBB
template <int dim>
struct Solid<dim>::PerTaskData_K
{
    FullMatrix<double>        cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_K(const unsigned int dofs_per_cell)
      :
      cell_matrix(dofs_per_cell, dofs_per_cell),
      local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      cell_matrix = 0.0;
    }
};

template <int dim>
struct Solid<dim>::ScratchData_K
{
    FEValues<dim> fe_values_ref;

    std::vector<std::vector<double> >                   Nx;
    std::vector<std::vector<Tensor<2, dim> > >          grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim> > > symm_grad_Nx;

    std::vector<std::vector<Tensor<1, dim> > >          grad_Nphi;

    ScratchData_K(const FiniteElement<dim> &fe_cell,
                  const QGauss<dim> &qf_cell,
                  const UpdateFlags uf_cell)
      :
      fe_values_ref(fe_cell, qf_cell, uf_cell),
      Nx(qf_cell.size(),
         std::vector<double>(fe_cell.dofs_per_cell)),
      grad_Nx(qf_cell.size(),
              std::vector<Tensor<2, dim> >(fe_cell.dofs_per_cell)),
      symm_grad_Nx(qf_cell.size(),
                   std::vector<SymmetricTensor<2, dim> >
                   (fe_cell.dofs_per_cell)),
      grad_Nphi(qf_cell.size(),
              std::vector<Tensor<1, dim> >(fe_cell.dofs_per_cell))
    {}

    ScratchData_K(const ScratchData_K &rhs)
      :
      fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags()),
      Nx(rhs.Nx),
      grad_Nx(rhs.grad_Nx),
      symm_grad_Nx(rhs.symm_grad_Nx),
      grad_Nphi(rhs.grad_Nphi)
    {}

    void reset()
    {
      const unsigned int n_q_points = Nx.size();
      const unsigned int n_dofs_per_cell = Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert( Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert( grad_Nx[q_point].size() == n_dofs_per_cell,
                  ExcInternalError());
          Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                  ExcInternalError());
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              Nx[q_point][k] = 0.0;
              grad_Nx[q_point][k] = 0.0;
              symm_grad_Nx[q_point][k] = 0.0;
              grad_Nphi[q_point][k] = 0.0;
            }
        }
    }

};

// Assembing tangent matrix
template <int dim>
void Solid<dim>::assemble_system_tangent()
{
    timer.enter_subsection("Assemble tangent matrix");
    std::cout << " ASM_K " << std::flush;

    tangent_matrix = 0.0;

    const UpdateFlags uf_cell(update_values    |
                              update_gradients |
                              update_JxW_values);

    PerTaskData_K per_task_data(dofs_per_cell);
    ScratchData_K scratch_data(fe, qf_cell, uf_cell);

    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::assemble_system_tangent_one_cell,
                    &Solid::copy_local_to_global_K,
                    scratch_data,
                    per_task_data);

    timer.leave_subsection();
}

template <int dim>
void Solid<dim>::copy_local_to_global_K(const PerTaskData_K &data)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        tangent_matrix.add(data.local_dof_indices[i],
                           data.local_dof_indices[j],
                           data.cell_matrix(i, j));
      }
    }  
}

template <int dim>
void Solid<dim>::assemble_system_tangent_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                  ScratchData_K &scratch,
                                                  PerTaskData_K &data)
{
    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const Tensor<2, dim> F_inv = lqph[q_point].get_F_inv();
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;

            if (k_group == u_dof)
              {
                scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
              }
            else if (k_group == phi_dof)
              {
                scratch.grad_Nphi[q_point][k] = scratch.fe_values_ref[phi_fe].gradient(k, q_point) * F_inv;
              }
            else if (k_group == p_dof)
              {
                scratch.Nx[q_point][k] = scratch.fe_values_ref[p_fe].value(k, q_point);
              }
            else if (k_group == J_dof)
              {
                scratch.Nx[q_point][k] = scratch.fe_values_ref[J_fe].value(k, q_point);
              }
            else
              Assert(k_group <= J_dof, ExcInternalError());
          }
      }

    // Extract some configuration dependent variables
    // from our QPH history objects for the current quadrature point.
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const Tensor<2, dim> sigma      = lqph[q_point].get_sigma();
        const SymmetricTensor<4, dim> c = lqph[q_point].get_c();
        const double detF               = lqph[q_point].get_J();
        const Tensor<1, dim> E_t        = lqph[q_point].get_E_t();
        const double d2Psi_vol_dJ2      = lqph[q_point].get_d2Psi_vol_dJ2();

        // Next define some aliases to make the assembly process easier to follow
        const std::vector<double> &N = scratch.Nx[q_point];
        const std::vector<SymmetricTensor<2, dim> > &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
        const std::vector<Tensor<2, dim> > &grad_Nx = scratch.grad_Nx[q_point];
        const std::vector<Tensor<1, dim> > &grad_Nphi = scratch.grad_Nphi[q_point];
        const double JxW = scratch.fe_values_ref.JxW(q_point) * detF;
        double epsilon = parameters.epsilon;

        for (unsigned int I = 0; I < dofs_per_cell; ++I)
          {
            const unsigned int component_i = fe.system_to_component_index(I).first;
            const unsigned int i_group     = fe.system_to_base_index(I).first.first;

            for (unsigned int J = 0; J <= I ; ++J)
              {
                const unsigned int component_j = fe.system_to_component_index(J).first;
                const unsigned int j_group     = fe.system_to_base_index(J).first.first;

                // Assemble the Kuu contribution
                if ((i_group == u_dof) && (j_group == u_dof))
                  {
                      // The material contribution:
                      data.cell_matrix(I, J) += symm_grad_Nx[I] * c
                                              * symm_grad_Nx[J] * JxW;
                      // geometrical stress contribution
                      if (component_i == component_j)
                          data.cell_matrix(I, J) += grad_Nx[I][component_i] * sigma
                                              * grad_Nx[J][component_j] * JxW;
                  }
                // The Kue contribution
                else if ((i_group == phi_dof) && (j_group == u_dof))
                  {
                        data.cell_matrix(I, J) -= ((grad_Nx[J][component_j] * E_t) * grad_Nphi[I][component_j]
                                                 + (grad_Nx[J][component_j] * grad_Nphi[I]) * E_t[component_j]
                                                 - (grad_Nphi[I] * E_t) * grad_Nx[J][component_j][component_j] )
                                                 * epsilon / parameters.phi_scal * JxW;
                  }
                else if ((i_group == u_dof) && (j_group == phi_dof))
                  {
                        data.cell_matrix(I, J) -= ((grad_Nx[I][component_i] * E_t) * grad_Nphi[J][component_i]
                                                 + (grad_Nx[I][component_i] * grad_Nphi[J]) * E_t[component_i] 
                                                 - (grad_Nphi[J] * E_t) * grad_Nx[I][component_i][component_i] )
                                                 * (epsilon / parameters.phi_scal) * JxW;
                  }

                // The Kee contribution
                else if ((i_group == phi_dof) && (j_group == phi_dof)) {
                    data.cell_matrix(I, J) -= grad_Nphi[I] * grad_Nphi[J]
                                             * epsilon / pow(parameters.phi_scal,2) * JxW;
                 }
                // The Kpu contribution
                else if ((i_group == p_dof) && (j_group == u_dof))
                  {
                    data.cell_matrix(I, J) += N[I] * (symm_grad_Nx[J] * StandardTensors<dim>::I)
                                              * JxW;
                  }
                // The Kjp contribution
                else if ((i_group == J_dof) && (j_group == p_dof))
                    data.cell_matrix(I, J) -= N[I] * N[J] * JxW;
                // The Kjj contribution
                else if ((i_group == J_dof) && (j_group == J_dof))
                    data.cell_matrix(I, J) += N[I] * d2Psi_vol_dJ2 * N[J] * JxW;
                else
                  Assert((i_group <= J_dof) && (j_group <= J_dof),
                         ExcInternalError());
              }
          }
      }

    // Copy the lower half of the local matrix into the upper half:
    for (unsigned int i = 0; i < dofs_per_cell; ++i) 
      for (unsigned int j = i + 1; j < dofs_per_cell; ++j) 
        data.cell_matrix(i, j) = data.cell_matrix(j, i);


}

/**********************************************************************************/

template <int dim>
struct Solid<dim>::PerTaskData_RHS
{
    Vector<double>            cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_RHS(const unsigned int dofs_per_cell)
      :
      cell_rhs(dofs_per_cell),
      local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      cell_rhs = 0.0;
    }
};

template <int dim>
struct Solid<dim>::ScratchData_RHS
{
    FEValues<dim>     fe_values_ref;
    FEFaceValues<dim> fe_face_values_ref;

    std::vector<std::vector<double> >                   Nx;
    std::vector<std::vector<SymmetricTensor<2, dim> > > symm_grad_Nx;
    std::vector<std::vector<Tensor<2, dim> > > grad_Nx;
    std::vector<std::vector<Tensor<1, dim> > > grad_Nphi;

    ScratchData_RHS(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
                    const QGauss<dim - 1> & qf_face, const UpdateFlags uf_face)
      :
      fe_values_ref(fe_cell, qf_cell, uf_cell),
      fe_face_values_ref(fe_cell, qf_face, uf_face),
      Nx(qf_cell.size(),
         std::vector<double>(fe_cell.dofs_per_cell)),
      symm_grad_Nx(qf_cell.size(),
                   std::vector<SymmetricTensor<2, dim> >
                   (fe_cell.dofs_per_cell)),
      grad_Nx(qf_cell.size(),
                   std::vector<Tensor<2, dim> >
                   (fe_cell.dofs_per_cell)),
      grad_Nphi(qf_cell.size(),
                   std::vector<Tensor<1, dim> >
                   (fe_cell.dofs_per_cell))
    {}

    ScratchData_RHS(const ScratchData_RHS &rhs)
      :
      fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags()),
      fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                         rhs.fe_face_values_ref.get_quadrature(),
                         rhs.fe_face_values_ref.get_update_flags()),
      Nx(rhs.Nx),
      symm_grad_Nx(rhs.symm_grad_Nx),
      grad_Nx(rhs.grad_Nx),
      grad_Nphi(rhs.grad_Nphi)
    {}

    void reset()
    {
      const unsigned int n_q_points      = symm_grad_Nx.size();
      const unsigned int n_dofs_per_cell = symm_grad_Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert( grad_Nphi[q_point].size() == n_dofs_per_cell, ExcInternalError());
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              Nx[q_point][k] = 0.0;
              symm_grad_Nx[q_point][k] = 0.0;
              grad_Nx[q_point][k] = 0.0;
              grad_Nphi[q_point][k] = 0.0;
            }
        }
    }

};

template <int dim>
void Solid<dim>::assemble_system_rhs()
{
    timer.enter_subsection("Assemble system right-hand side");
    std::cout << " ASM_R " << std::flush;

    system_rhs = 0.0;

    const UpdateFlags uf_cell(update_values |
                              update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values |
                              update_normal_vectors |
                              update_JxW_values);

    PerTaskData_RHS per_task_data(dofs_per_cell);
    ScratchData_RHS scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face);

    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::assemble_system_rhs_one_cell,
                    &Solid::copy_local_to_global_rhs,
                    scratch_data,
                    per_task_data);

    timer.leave_subsection();

}

template <int dim>
void Solid<dim>::copy_local_to_global_rhs(const PerTaskData_RHS &data)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      system_rhs(data.local_dof_indices[i]) += data.cell_rhs(i);
}

template <int dim>
void Solid<dim>::assemble_system_rhs_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           ScratchData_RHS &scratch,
                                           PerTaskData_RHS &data)
{
    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const Tensor<2, dim> F_inv = lqph[q_point].get_F_inv();

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;

            if (k_group == u_dof) {
                scratch.symm_grad_Nx[q_point][k]
                    = symmetrize(scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv);
                scratch.grad_Nx[q_point][k]
                    = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
            }
            else if (k_group == phi_dof)
                scratch.grad_Nphi[q_point][k] = scratch.fe_values_ref[phi_fe].gradient(k, q_point) * F_inv;
            else if (k_group == p_dof)
                scratch.Nx[q_point][k]
                    = scratch.fe_values_ref[p_fe].value(k, q_point);
            else if (k_group == J_dof)
                scratch.Nx[q_point][k]
                    = scratch.fe_values_ref[J_fe].value(k, q_point);
            else
                Assert(k_group <= J_dof, ExcInternalError());
          }
      }

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const SymmetricTensor<2, dim> sigma = lqph[q_point].get_sigma();
        const double detF        = lqph[q_point].get_J();
        const Tensor<1, dim> E_t = lqph[q_point].get_E_t();
        const double J_tilde     = lqph[q_point].get_J_tilde();
        const double p_tilde     = lqph[q_point].get_p_tilde();
        const double dPsi_vol_dJ = lqph[q_point].get_dPsi_vol_dJ();

        const std::vector<double> &N = scratch.Nx[q_point];
        const std::vector<SymmetricTensor<2, dim> > &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
        const std::vector<Tensor<1, dim> > &grad_Nphi = scratch.grad_Nphi[q_point];
        const double JxW = scratch.fe_values_ref.JxW(q_point) * detF;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int i_group = fe.system_to_base_index(i).first.first;

            if (i_group == u_dof) {
              data.cell_rhs(i) -= (symm_grad_Nx[i] * sigma) * JxW;
            }
            else if (i_group == phi_dof) {
              data.cell_rhs(i) -= grad_Nphi[i] * parameters.epsilon / parameters.phi_scal * E_t * JxW;
            }
            else if (i_group == p_dof) {
              data.cell_rhs(i) -= N[i] * (detF - J_tilde) * JxW;
            }
            else if (i_group == J_dof) {
              data.cell_rhs(i) -= N[i] * (dPsi_vol_dJ - p_tilde) * JxW;
            }
            else
              Assert(i_group <= J_dof, ExcInternalError());

          }
      }

    // Assemble the Neumann contribution.
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
        if (cell->face(face)->at_boundary() == true
            && cell->face(face)->boundary_id() == 2) {   // Select the surface to apply charge
            scratch.fe_face_values_ref.reinit(cell, face);

            for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point) {
                const double time_ramp = (time.current() / time.end());
                const double charge_density = parameters.charge_density * time_ramp;
                const double JxW = scratch.fe_face_values_ref.JxW(f_q_point);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int i_group = fe.system_to_base_index(i).first.first;
                    if (i_group == phi_dof) {
                        const double N = scratch.fe_face_values_ref.shape_value(i, f_q_point);

                        data.cell_rhs(i) -= N * charge_density * JxW / sqrt(parameters.epsilon);
                    }
                }
            }
        } //Finish assemble potential neumann contribution

    }

}

/**********************************************************************************/
//Define the potential essential boundary condition
template <int dim>
class PotentialFunction : public Function<dim>
{
    public:
    PotentialFunction (unsigned int n_components, Time &time, 
                        Parameters::AllParameters &parameters,
                        Geometry &geo) : 
        Function<dim>(n_components),
        time(time),
        n_components(n_components),
        parameters(parameters),
        geo(geo)
    {}
    virtual void vector_value (const Point<dim> &p,
                               Vector<double> &values) const;
    Time time;
    unsigned int n_components;
    const Parameters::AllParameters &parameters;
    const Geometry &geo;
};


template <int dim>
void PotentialFunction<dim>::vector_value (const Point<dim> &/*p*/,
                              Vector<double> &values) const
{
    Assert (values.size() == n_components,
            ExcDimensionMismatch(values.size(), n_components) );

    const int steps = time.get_total_timestep();
    const double dv = parameters.total_voltage / steps;
    
    values = 0.0;
    values[3] = dv * parameters.phi_scal;
}

template <int dim>
void Solid<dim>::make_constraints(const int &it_nr)
{
    // Essential boundary condition
    std::cout << " CST " << std::flush;

    // Since the constraints are different at different Newton iterations, we
    // need to clear the constraints matrix and completely rebuild
    // it. However, after the first iteration, the constraints remain the same
    // and we can simply skip the rebuilding step if we do not clear it.
    if (it_nr > 1)
        return;
    constraints.clear();
    const bool apply_dirichlet_bc = (it_nr == 0);

    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    const FEValuesExtractors::Scalar z_displacement(2);
    const FEValuesExtractors::Scalar phi_value(3);

    {
        const int boundary_id = 2;
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement) |
                                                 fe.component_mask(y_displacement));
    }

    constraints.add_lines(leftmid_zdof);

// Zero potential at top-plane
    {
        const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(phi_value));
    }
// Zero potential at mid-plane
    {
      const int boundary_id = 7;
            VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(phi_value));
    }

//Plane strain constrain
    {
      const int boundary_id = 3;
            VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(y_displacement));
    }
    {
      const int boundary_id = 5;
            VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(y_displacement));
    }

// Apply voltage at bottom plane
    {
        const int boundary_id = 6;
        if (apply_dirichlet_bc == 1)
            VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                     boundary_id,
                                                     PotentialFunction<dim>(n_components, time,
                                                                    parameters, geo),
                                                     constraints,
                                                     fe.component_mask(phi_value) );
        else
            VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(phi_value) );
    }
    constraints.close();
}


/**********************************************************************************/
// Perform static condensation
template <int dim>
struct Solid<dim>::PerTaskData_SC
{
    FullMatrix<double>      cell_matrix;
    std::vector<types::global_dof_index>    local_dof_indices;
    
    FullMatrix<double>      k_orig;
    FullMatrix<double>      k_pu;
    FullMatrix<double>      k_pJ;
    FullMatrix<double>      k_JJ;
    FullMatrix<double>      k_pJ_inv;
    FullMatrix<double>      k_bbar;
    FullMatrix<double>      A;
    FullMatrix<double>      B;
    FullMatrix<double>      C;

    PerTaskData_SC (const unsigned int dofs_per_cell,
                    const unsigned int n_xi,
                    const unsigned int n_p,
                    const unsigned int n_J)
    :
    cell_matrix(dofs_per_cell, dofs_per_cell),
    local_dof_indices(dofs_per_cell),
    k_orig(dofs_per_cell, dofs_per_cell),
    k_pu(n_p, n_xi),
    k_pJ(n_p, n_J),
    k_JJ(n_J, n_J),
    k_pJ_inv(n_p, n_J),
    k_bbar(n_xi, n_xi),
    A(n_J, n_xi),
    B(n_J, n_xi),
    C(n_p, n_xi)
    {}

    void reset()
    {}
};

template <int dim>
struct Solid<dim>::ScratchData_SC
{
    void reset()
    {}
};


template <int dim>
void Solid<dim>::assemble_sc()
{
    timer.enter_subsection("Perform static condensation");
    std::cout << " ASM_SC " << std::flush;
    PerTaskData_SC per_task_data(dofs_per_cell, 
                                 element_indices_xi.size(),
                                 element_indices_p.size(),
                                 element_indices_J.size());
    ScratchData_SC scratch_data;

    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::assemble_sc_one_cell,
                    &Solid::copy_local_to_global_sc,
                    scratch_data,
                    per_task_data);


    timer.leave_subsection();
}

template <int dim>
void Solid<dim>::copy_local_to_global_sc(const PerTaskData_SC &data)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
        tangent_matrix.add(data.local_dof_indices[i],
                           data.local_dof_indices[j],
                           data.cell_matrix(i, j));
}

template <int dim>
void Solid<dim>::assemble_sc_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                      ScratchData_SC &scratch,
                                      PerTaskData_SC &data)
{
    data.reset();
    scratch.reset();
    cell->get_dof_indices(data.local_dof_indices);

    data.k_orig.extract_submatrix_from(tangent_matrix,
                                       data.local_dof_indices,
                                       data.local_dof_indices);

    data.k_pu.extract_submatrix_from(data.k_orig,
                                     element_indices_p,
                                     element_indices_xi);

    data.k_pJ.extract_submatrix_from(data.k_orig,
                                     element_indices_p,
                                     element_indices_J);

    data.k_JJ.extract_submatrix_from(data.k_orig,
                                     element_indices_J,
                                     element_indices_J);

    data.k_pJ_inv.invert(data.k_pJ);
    data.k_pJ_inv.mmult(data.A, data.k_pu);
    data.k_JJ.mmult(data.B, data.A);
    data.k_pJ_inv.Tmmult(data.C, data.B);
    data.k_pu.Tmmult(data.k_bbar, data.C);

    data.k_bbar.scatter_matrix_to(element_indices_xi,
                                  element_indices_xi,
                                  data.cell_matrix);

    data.k_pJ_inv.add(-1.0, data.k_pJ);
    data.k_pJ_inv.scatter_matrix_to(element_indices_p,
                                    element_indices_J,
                                    data.cell_matrix);
}


/**********************************************************************************/
//Assemble mass matrix
template <int dim>
void Solid<dim>::assemble_mass_matrix()
{
    mass_matrix = 0;
    FEValues<dim> fe_values (fe, qf_cell,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

    unsigned int dofs_per_cell = fe.dofs_per_cell;
    unsigned int n_q_points = qf_cell.size();

    FullMatrix<double>   cell_mass_matrix (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active(),
                                                   endc = dof_handler_ref.end();
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell_mass_matrix = 0.0;
        
        for (unsigned int q=0; q<n_q_points; ++q) {
            for (unsigned int i=0; i<dofs_per_cell; i++) {
                const unsigned int i_group     = fe.system_to_base_index(i).first.first;

                for (unsigned int j=0; j<dofs_per_cell; j++) {
                    const unsigned int j_group     = fe.system_to_base_index(j).first.first;

                    if ((i_group == u_dof) && (j_group == u_dof))
                        cell_mass_matrix[i][j] += fe_values.shape_value(i,q) *
                                                  fe_values.shape_value(j,q) *
                                                  fe_values.JxW(q) * parameters.rho;
                }
            }
        }
        
        cell -> get_dof_indices (local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            mass_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_mass_matrix(i, j));
          }
        }  
    }
}

/**********************************************************************************/
template <int dim>
void Solid<dim>::system_setup()
{
    timer.enter_subsection("Setup system");

    //Initialize block component
    std::vector<unsigned int> block_component(n_components, xi_blk);
    block_component[p_component] = p_blk;
    block_component[J_component] = J_blk;

    dof_handler_ref.distribute_dofs (fe);

    //Renumbering, arrange the pressure dof to block
    DoFRenumbering::Cuthill_McKee (dof_handler_ref);
    DoFRenumbering::component_wise (dof_handler_ref, block_component);
    
    DoFTools::count_dofs_per_block (dof_handler_ref, dofs_per_block, block_component);

    //Extract the u and phi index
    total_indices_u.clear();
    total_indices_phi.clear();
    for (unsigned int i=0;i<dofs_per_block[xi_blk];i++) {
        unsigned int idx = i % (phi_component+1);
        if (idx == 3)
            total_indices_phi.push_back(i);
        else
            total_indices_u.push_back(i);

    }

    //Define probe vertex for post-processing. This is for bending problem.
    for (typename DoFHandler<dim>::active_cell_iterator 
                  cell = dof_handler_ref.begin_active();
                  cell != dof_handler_ref.end(); ++cell)  {
        if (cell->is_locally_owned()) {
            for (unsigned int v=0; v< GeometryInfo<dim>::vertices_per_cell; ++v) {
                if ( (abs(cell->vertex(v)[0] - geo.x) < 1e-8) &&
                     (abs(cell->vertex(v)[1] - geo.y) < 1e-8) &&
                     (abs(cell->vertex(v)[2] - geo.z) < 1e-8) ) {
                    probe_vertex_x = cell->vertex_dof_index(v, first_u_component);
                    probe_vertex_z = cell->vertex_dof_index(v, first_u_component+2);
                }

                if ( (abs(cell->vertex(v)[0] - geo.x) < 1e-8) &&
                     (abs(cell->vertex(v)[1] - geo.y) < 1e-8) &&
                     (abs(cell->vertex(v)[2] - geo.z/2) < 1e-8) ) {
                    probe_vertex_x_2 = cell->vertex_dof_index(v, first_u_component);
                    probe_vertex_z_2 = cell->vertex_dof_index(v, first_u_component+2);

                }

                if ( (abs(cell->vertex(v)[0] - geo.x) < 1e-8) &&
                     (abs(cell->vertex(v)[1] - geo.y) < 1e-8) &&
                     (abs(cell->vertex(v)[2]) < 1e-8) ) {
                    probe_vertex_phi = cell->vertex_dof_index(v, phi_component);

                }

                // Define left middle point set
                if ( (abs(cell->vertex(v)[0]) < 1e-8) &&
                     (abs(cell->vertex(v)[2] - geo.z/2) < 1e-8) ) {
                     leftmid_zdof.insert(cell->vertex_dof_index(v, first_u_component+2));
                }
            }
        }
    }

    // Output mesh summary
    std::cout << "Mesh summary: " << std::endl
              << "\t Number of active cells: " << triangulation.n_active_cells() << std::endl
              << "\t Number of degrees of freedom: " << dof_handler_ref.n_dofs() << std::endl;

    // Setup the sparsity pattern and tangent matrix
    tangent_matrix.clear();
    mass_matrix.clear();
    {
        const types::global_dof_index n_dofs_xi = dofs_per_block[xi_blk];
        const types::global_dof_index n_dofs_p = dofs_per_block[p_blk];
        const types::global_dof_index n_dofs_J = dofs_per_block[J_blk];
        std::cout << "\t\t ---- Number of displacement & phi dofs: " << n_dofs_xi << std::endl
                  << "\t\t ---- Number of pressure           dofs: " << n_dofs_p << std::endl
                  << "\t\t ---- Number of J                  dofs: " << n_dofs_J << std::endl;
        
        BlockDynamicSparsityPattern dsp(n_blocks, n_blocks);
        dsp.block(xi_blk, xi_blk).reinit(n_dofs_xi,  n_dofs_xi);
        dsp.block(xi_blk, p_blk ).reinit(n_dofs_xi,  n_dofs_p);
        dsp.block(xi_blk, J_blk ).reinit(n_dofs_xi,  n_dofs_J);

        dsp.block(p_blk,  xi_blk).reinit(n_dofs_p,   n_dofs_xi);
        dsp.block(p_blk,  p_blk ).reinit(n_dofs_p,   n_dofs_p);
        dsp.block(p_blk,  J_blk ).reinit(n_dofs_p,   n_dofs_J);

        dsp.block(J_blk,  xi_blk).reinit(n_dofs_J,   n_dofs_xi);
        dsp.block(J_blk,  p_blk ).reinit(n_dofs_J,   n_dofs_p);
        dsp.block(J_blk,  J_blk ).reinit(n_dofs_J,   n_dofs_J);
        dsp.collect_sizes();
        
        Table<2, DoFTools::Coupling> coupling(n_components, n_components);
        for (unsigned int ii=0; ii<n_components; ++ii) {
            for (unsigned int jj=0; jj<n_components; ++jj) {
                coupling[ii][jj] = DoFTools::always;
            }
        }
                
        DoFTools::make_sparsity_pattern(dof_handler_ref,
                                        coupling,
                                        dsp,
                                        constraints,
                                        false);
        sparsity_pattern.copy_from(dsp);
    }

    tangent_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);

    assemble_mass_matrix();

    system_rhs.reinit (dofs_per_block);
    system_rhs.collect_sizes();
    solution_n.reinit (dofs_per_block);
    solution_n.collect_sizes();

    solution_n_loading.reinit (dofs_per_block);
    solution_n_loading.collect_sizes();

    setup_qph();

    timer.leave_subsection();
}


/**********************************************************************************/
template <int dim>
void Solid<dim>::solve_nonlinear_timestep(BlockVector<double> &solution_delta, Time &time)
{
    if (!time.is_pre())
        std::cout << std::endl << "Timestep: " << time.get_timestep() << " @ "
                  << time.current() << " s" << std::endl;
    else
        std::cout << std::endl << "Pre-stretch Timestep: " << time.get_timestep() << " @ "
                  << time.current() << " s" << std::endl;

    BlockVector<double> newton_update(dofs_per_block);

    // Initiate errors in the current time step
    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_msg_header();

    
    for (unsigned int newton_itr = 0; 
         newton_itr < parameters.max_iterations_NR; 
         ++newton_itr) {
        std::cout << " " << std::setw(2) << newton_itr << " " << std::flush;
        
        tangent_matrix = 0.0;
        system_rhs = 0.0;
        
        assemble_system_rhs();

        // Get the residuals
        get_error_residual(error_residual);
        if (newton_itr == 1)
            error_residual_0 = error_residual;
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        // Check whether converged
        if (newton_itr > 1 && error_residual_norm.u <= parameters.tol_f &&
                              error_update_norm.u <= parameters.tol_u)
        {
            std::cout << "    ---- CONVERGED! " << std::endl;
            print_msg_footer();
            break;
        }

        // Continue iteration
        assemble_system_tangent();
        make_constraints(newton_itr);

        constraints.condense(tangent_matrix, system_rhs);

        // Solve linear system equations
        const std::pair<unsigned int, double>
            lin_solver_output = solve_linear_system(newton_update);

        // Recover the scaled potential
        for (unsigned int i=0;i<dofs_per_block[xi_blk];i++) {
            if (i % (dim + 1) == 3)
                newton_update.block(xi_blk)[i] /= parameters.phi_scal;
        }

        // Check errors
        get_error_update(newton_update, error_update);
        if (newton_itr == 0)
            error_update_0 = error_update;
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        // Get solution
        solution_delta += newton_update;
        update_qph_incremental(solution_delta);

        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                  << std::scientific << lin_solver_output.first << "     "  //linear solver iteration number
                  << lin_solver_output.second << "    "  //linear solver residuals
                  << error_residual_norm.norm << "    "
                  << error_residual_norm.u    << "    "
                  << error_residual_norm.phi  << "    "
                  << error_residual_norm.p    << "    "
                  << error_residual_norm.J    << "    "
                  << std::endl;
    }
}


template <int dim>
std::pair<unsigned int, double>
Solid<dim>::solve_linear_system(BlockVector<double> &newton_update)
{
    unsigned int lin_it = 0;
    double lin_res = 0.0;

    assemble_sc();

    timer.enter_subsection("Linear solver");
    std::cout << " SLV " << std::flush;

    BlockVector<double> A(dofs_per_block);
    BlockVector<double> B(dofs_per_block);
   
    {
        // Assemble rhs for SC
        tangent_matrix.block(p_blk, J_blk).vmult(A.block(J_blk), system_rhs.block(p_blk));
        tangent_matrix.block(J_blk, J_blk).vmult(B.block(J_blk), A.block(J_blk));

        A.block(J_blk) = system_rhs.block(J_blk);
        A.block(J_blk) -= B.block(J_blk);

        tangent_matrix.block(p_blk, J_blk).Tvmult(A.block(p_blk), A.block(J_blk));
        tangent_matrix.block(xi_blk, p_blk).vmult(A.block(xi_blk), A.block(p_blk));

        system_rhs.block(xi_blk) -= A.block(xi_blk);

        if (parameters.type_lin == "GMRES") {
          TrilinosWrappers::SparseMatrix tm;
          TrilinosWrappers::MPI::Vector rhs;
          TrilinosWrappers::MPI::Vector nu;
          tm.reinit(tangent_matrix.block(xi_blk, xi_blk), 0);
          rhs = system_rhs.block(xi_blk);
          nu = newton_update.block(xi_blk);
            
          {
            Epetra_Vector x(View, tm.domain_partitioner(), nu.begin());
            Epetra_Vector b(View, tm.domain_partitioner(), rhs.begin());
            AztecOO solver;
            solver.SetAztecOption(AZ_output, AZ_none);
            solver.SetAztecOption(AZ_solver, AZ_gmres);
            solver.SetRHS(&b);
            solver.SetLHS(&x);

            solver.SetAztecOption(AZ_precond,         AZ_dom_decomp);
            solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
            solver.SetAztecOption(AZ_overlap, 1);
            solver.SetAztecOption(AZ_reorder, 1);

            solver.SetAztecParam(AZ_drop,      0);
            solver.SetAztecParam(AZ_ilut_fill, 5);
            solver.SetAztecParam(AZ_athresh,   0);
            solver.SetAztecParam(AZ_rthresh,   0);

            solver.SetUserMatrix(const_cast<Epetra_CrsMatrix *>
                                    (&tm.trilinos_matrix()));

            const int solver_its = tangent_matrix.block(xi_blk, xi_blk).m();
            const double tol_sol = 1e-12;
            solver.Iterate(solver_its, tol_sol);
            lin_it = solver.NumIters();
            lin_res = solver.TrueResidual();

            newton_update.block(xi_blk) = nu;
          }

      }
      else if (parameters.type_lin == "Direct") {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(tangent_matrix.block(xi_blk, xi_blk));
        A_direct.vmult(newton_update.block(xi_blk), system_rhs.block(xi_blk));
        lin_it = 1;
        lin_res = 0.0;
      }
      else
        Assert (false, ExcMessage("Linear solver type not implemented"));
    }
    constraints.distribute(newton_update);
    timer.leave_subsection();

    // Begin post processing to get J and p    
    timer.enter_subsection("Linear solver postprocessing");

    std::cout << " PP " << std::flush;
    //Post-processing solve J
    {
        tangent_matrix.block(p_blk, xi_blk).vmult(A.block(p_blk), newton_update.block(xi_blk));
        A.block(p_blk) *= -1.0;
        A.block(p_blk) += system_rhs.block(p_blk);
        tangent_matrix.block(p_blk, J_blk).vmult(newton_update.block(J_blk),
                                                 A.block(p_blk));
    }
    constraints.distribute(newton_update);

    //Post-processing solve p
    {
        tangent_matrix.block(J_blk, J_blk).vmult(A.block(J_blk),
                                                 newton_update.block(J_blk));
        A.block(J_blk) *= -1.0;
        A.block(J_blk) += system_rhs.block(J_blk);
        tangent_matrix.block(p_blk, J_blk).Tvmult(newton_update.block(p_blk),
                                                  A.block(J_blk));
    }
    constraints.distribute(newton_update);

    timer.leave_subsection();

    return std::make_pair(lin_it, lin_res);    
}

/**********************************************************************************/
template <int dim>
void Solid<dim>::print_msg_header()
{
    static const unsigned int l_width = 148;
    
    for (unsigned int i=0;i<l_width;++i)
        std::cout << "-";
    std::cout << std::endl;
    
    std::cout << "                 SOLVER STEP                  "
              << " |    LIN_IT    LIN_RES      RES_NORM      "
              << " RES_U       RES_PHI       RES_p    "
              << " RES_J" << std::endl;

    for (unsigned int i=0;i<l_width;++i)
        std::cout << "-";
    std::cout << std::endl;
}

template <int dim>
void Solid<dim>::print_msg_footer()
{
    static const unsigned int l_width = 148;
    
    for (unsigned int i=0;i<l_width;++i)
        std::cout << "-";
    std::cout << std::endl;
}

template <int dim>
BlockVector<double> Solid<dim>::get_total_solution(const BlockVector<double> &solution_delta) const
{
    BlockVector<double> solution_total(solution_n);
    solution_total += solution_delta;
    return solution_total;
}

/**********************************************************************************/
// Determine the true residual error
template <int dim>
void Solid<dim>::get_error_residual(Errors &error_residual)
{
    unsigned int ndof = dof_handler_ref.n_dofs();
    BlockVector<double> error_res(dofs_per_block);

    for (unsigned int i=0;i<ndof;++i)
        if (!constraints.is_constrained(i))
            error_res(i) = system_rhs(i);

    std::vector<double> vector_u(total_indices_u.size());
    std::vector<double> vector_phi(total_indices_phi.size());

    error_res.block(xi_blk).extract_subvector_to(total_indices_u, vector_u);
    error_res.block(xi_blk).extract_subvector_to(total_indices_phi, vector_phi);
    error_residual.u = get_l2_norm(vector_u);
    error_residual.phi = get_l2_norm(vector_phi);
    
    error_residual.norm = error_res.l2_norm();
    error_residual.p = error_res.block(p_blk).l2_norm();
    error_residual.J = error_res.block(J_blk).l2_norm();
}

// Determine the true Newton update error for the problem
template <int dim>
void Solid<dim>::get_error_update(const BlockVector<double> &newton_update,
                                  Errors &error_update)
{
    unsigned int ndof = dof_handler_ref.n_dofs();
    BlockVector<double> error_ud(dofs_per_block);

    for (unsigned int i=0;i<ndof;++i)
        if (!constraints.is_constrained(i))
            error_ud(i) = newton_update(i);

    std::vector<double> vector_u(total_indices_u.size());
    std::vector<double> vector_phi(total_indices_phi.size());

    error_ud.block(xi_blk).extract_subvector_to(total_indices_u, vector_u);
    error_ud.block(xi_blk).extract_subvector_to(total_indices_phi, vector_phi);
    error_update.u = get_l2_norm(vector_u);
    error_update.phi = get_l2_norm(vector_phi);

    
    error_update.norm = error_ud.l2_norm();
    error_update.p = error_ud.block(p_blk).l2_norm();
    error_update.J = error_ud.block(J_blk).l2_norm();
}

template <int dim>
double Solid<dim>::get_l2_norm(const std::vector<double> &vec)
{
    double l2 = 0.0;
    for (unsigned int i=0;i<vec.size();++i)
        l2 += vec[i]*vec[i];
    return sqrt(l2);
}



/**********************************************************************************/
// Generate meshes
template <int dim>
void Solid<dim>::make_grid()
{
    double scale = parameters.scale;

    // Read grid file for beam
    GridIn<dim> grid_in;
    grid_in.attach_triangulation (triangulation);
    std::ifstream input_file(parameters.inp_file_name);
    grid_in.read_ucd (input_file);
    GridTools::scale(scale, triangulation);

    // Assign boundary indicator
    int count = 0;
    for (auto cell : triangulation.active_cell_iterators()) {
        for (unsigned int face = 0; face < 6; face++) {
             //face < GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary() == true
                && fabs(cell->face(face)->center()[2] - geo.z) < 1e-4*scale) 
                cell->face(face)->set_boundary_id(1);
            else if (cell->face(face)->at_boundary() == true
                && fabs(cell->face(face)->center()[2]) < 1e-4*scale) 
                cell->face(face)->set_boundary_id(6);
            else if (cell->face(face)->at_boundary() == true
                && fabs(cell->face(face)->center()[0]) < 1e-4*scale) 
                cell->face(face)->set_boundary_id(2);
            else if (cell->face(face)->at_boundary() == true
                && fabs(cell->face(face)->center()[1]) < 1e-4*scale) 
                cell->face(face)->set_boundary_id(3);
            else if (cell->face(face)->at_boundary() == true
                && fabs(cell->face(face)->center()[0] - geo.x) < 1e-4*scale) 
                cell->face(face)->set_boundary_id(4);
            else if (cell->face(face)->at_boundary() == true
                && fabs(cell->face(face)->center()[1] - geo.y) < 1e-4*scale) 
                cell->face(face)->set_boundary_id(5);
            if (cell->face(face)->at_boundary() == false
                && fabs(cell->face(face)->center()[2] - geo.z/2.0) < 1e-4*scale) {

                cell->face(face)->set_boundary_id(7);
                count++;
                
            }
              
        }
    }   

    // Global refinement
    for (unsigned int step=0; step<parameters.global_refinement; step++) {
      typename Triangulation<dim>::active_cell_iterator 
        cell = triangulation.begin_active(), endc = triangulation.end();
      for (; cell != endc; ++cell)
        cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
    }

}





template <int dim>
void Solid<dim>::output_results(Time &time) const
{
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> solution_name(dim, "displacement");
    solution_name.push_back("potential");
    solution_name.push_back("pressure");
    solution_name.push_back("J_tilde");

    data_out.attach_dof_handler(dof_handler_ref);

    if (time.is_pre())
        data_out.add_data_vector(solution_n,
                                 solution_name,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);
    else
        data_out.add_data_vector(solution_n_loading,
                                 solution_name,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);


    Vector<double> mises_of_stress (triangulation.n_active_cells());
    Vector<double> detF_vec (triangulation.n_active_cells());
    {
        unsigned int index = 0;
        for (auto cell : triangulation.active_cell_iterators()) {
            SymmetricTensor<2, dim> as;
            double detf = 0.0;
            for (unsigned int q=0; q<qf_cell.size(); ++q) {
                as += reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].get_sigma();
                detf +=reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].get_J(); 
            }
                
            as = as / qf_cell.size();
            double mises = 0.5 * (pow((as[0][0] - as[1][1]),2) + 
                                  pow((as[0][0] - as[2][2]),2) + 
                                  pow((as[2][2] - as[1][1]),2) +
                                  6.0*(as[1][2]*as[1][2] + as[2][0]*as[2][0] + as[0][1]*as[0][1]));
            mises_of_stress(index) = sqrt(mises);
            detF_vec(index) = detf / qf_cell.size();
            index++;
        }
    }
    
    data_out.add_data_vector(mises_of_stress, "Mises Stress");
    data_out.add_data_vector(detF_vec, "detF");

    Vector<double> soln(solution_n.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
        soln(i) = solution_n(i);
    MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);
    data_out.build_patches(q_mapping, degree);

    std::ostringstream filename;
    if (!time.is_pre())
        filename << "solution-" << time.get_timestep() << ".vtk";
    else
        filename << "solution-pre-" << time.get_timestep() << ".vtk";


    std::cout << "Output to solution file: " << filename.str() << std::endl;

    std::ofstream output(filename.str().c_str());
    data_out.write_vtk(output);

}


template <int dim>
void Solid<dim>::run()
{

    make_grid();
    system_setup();
    //Initialize the dialation J value
    {
        ConstraintMatrix constraints;
        constraints.close();
        const ComponentSelectFunction<dim>
        J_mask (J_component, n_components);

        VectorTools::project (dof_handler_ref,
                              constraints,
                              QGauss<dim>(degree+2),
                              J_mask,
                              solution_n);
    }
    output_results(time_pre);
    solution_n_loading = solution_n;

    BlockVector<double> solution_delta(dofs_per_block);

    // Pre-stretch relaxation
    PRES = StandardTensors<dim>::I;
    std::cout << "****************************" << std::endl;
    std::cout << "Begin Pre-stretch relaxation" << std::endl;
    std::cout << "****************************" << std::endl;
    while (fabs(time_pre.current() - time_pre.end()) > time_pre.get_dt()/10.0)
    {
        time_pre.increment();

        if (time_pre.current() < 1.0) {
            PRES[0][0] = pow(parameters.pre_x, time_pre.current());
            PRES[1][1] = pow(parameters.pre_y, time_pre.current());
            PRES[2][2] = pow(parameters.pre_z, time_pre.current());
        }
        else {
            PRES[0][0] = parameters.pre_x;
            PRES[1][1] = parameters.pre_y;
            PRES[2][2] = parameters.pre_z;
        }

        solution_delta = 0.0;
        solve_nonlinear_timestep(solution_delta, time_pre);
        solution_n += solution_delta;
        output_results(time_pre);
    }

    std::cout << "*******************************************************" << std::endl;
    std::cout << "Finish pre-stretching relaxation. Begin applying force." << std::endl;
    std::cout << "*******************************************************" << std::endl;
    sleep(1);

    std::ofstream recfile("node_probe_rec.dat");
    recfile << 0.0 << ", " << 0.0 << std::endl;

    // Applying loading
    while (fabs(time.current() - time.end()) > (time.get_dt()/10.0))
    {
        time.increment();

        solution_delta = 0.0;
        solve_nonlinear_timestep(solution_delta, time);
        solution_n += solution_delta;
        solution_n_loading += solution_delta;

        output_results(time);

        // store the probe vertex
        const unsigned int idx = time.get_timestep();
        probe_node_potential[idx] = solution_n_loading[probe_vertex_phi];
        double x1 = solution_n[probe_vertex_x];
        double z1 = solution_n[probe_vertex_z];
        double x2 = solution_n[probe_vertex_x_2];
        double z2 = solution_n[probe_vertex_z_2];
        double angle = atan2(abs(x1-x2), (geo.z/2.0+z1-z2));

        recfile << probe_node_potential[idx] << ", " << setprecision(16) << angle << std::endl;
    }

    recfile.close();
}


}


int main (int argc, char *argv[])
{
  try
    {
      deallog.depth_console(0);

	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
									     numbers::invalid_unsigned_int);

      deprog::Solid<3> solid_3d("parameters.prm");
      solid_3d.run();

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

