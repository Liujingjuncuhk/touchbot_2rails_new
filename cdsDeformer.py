import numpy as np
import pyvista as pv
from scipy.linalg import ldl, solve_triangular
from cableDrivenSurface import cableDrivenSurface
from utilities import * 
# from lemkelcp import lemketableau
from qpsolvers import solve_qp
from scipy.optimize import minimize
import time

class cdsDeformer:
    def __init__(self, cds: cableDrivenSurface):
        self.cds = cds
        self.N44 = np.eye(4) - 1/4*np.ones((4, 4))
        self.N22 = np.eye(2) - 1/2*np.ones((2, 2))
        self.N1212 = np.zeros((12, 12))
        for i in range(4):
            for j in range(3):
                for k in range(4):
                    if k==i:
                        self.N1212[3*i+j, 3*k+j] = 3.0/4.0
                    else:
                        self.N1212[3*i+j, 3*k+j] = -1.0/4.0
        self.matA_tilde, self.matA_tilde_list, self.matAT_tilde_list, self.matATA_tilde_list, self.VecB2Add= self.construct_matA()
        self.matAT_tilde = self.matA_tilde.T
        self.matATA_tilde = self.matAT_tilde @ self.matA_tilde
        self.stiffness_arm = 50
  
    # FK part

    def fk_length_startMesh(self, target_cable_length, starting_vertices):
        Q_a = np.zeros((3*self.cds.num_vert, 1))
        for i in range(self.cds.num_vert):
            Q_a[3*i:3*i+3, 0] = starting_vertices[i]
        Q_ad = np.zeros((3*self.cds.num_vert, 1))
        Q_a_last = Q_a.copy()
        total_time = 5.0
        h = 0.1
        t_a = 0.0
        phi_Qfree = np.zeros((self.cds.nCable, 1))
        H_free = np.zeros((self.cds.nCable, 3*self.cds.num_vert))
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + h * self.cds.W_mat @ K_mat 
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.cds.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.cds.gravity_vec)
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.cds.num_vert, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            for i in range(self.cds.nCable):
                idx_pp = self.cds.pp_idx[i]
                unit_vec = (self.cds.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                phi_Qfree[i] = target_cable_length[i] - np.linalg.norm(self.cds.pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
            lcp_Mmat = h * H_free @ self.cds.W_mat @ A_inv @ H_free.T
            # check if lcp_Mmat is PD
            M_is_PD = np.all(np.linalg.eigvals(lcp_Mmat) > 0)
            lcp_q = phi_Qfree.reshape((self.cds.nCable,))
            lcp_sol = lcp.lemkelcp(lcp_Mmat, lcp_q)
            if not lcp_sol[1] == 0:
                print("lcp failed: ")
                break
            cable_tension = lcp_sol[0]
            # print("cur time: ", t_a, "cable_tension: ", cable_tension, "lcp_q: ", lcp_q.flatten(), "M_is_PD: ", M_is_PD)
            # cable_tension = solve_qp(lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.cds.nCable,)), solver = 'cvxopt')
            cable_tension.reshape((self.cds.nCable, 1))
            dv_cor = A_inv @ (self.cds.W_mat @ H_free.T @ cable_tension).reshape((3*self.cds.num_vert, 1))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            if np.linalg.norm(Q_a - Q_a_last)/self.cds.num_vert < 1e-6:
                break
        vert_length = self.cds.cur_vertices.copy()
        for i in range(self.cds.num_vert):
            vert_length[i] = Q_a[3*i:3*i+3].reshape((3,))
        return vert_length, cable_tension.flatten(), R_list_1212

    def fk_length_force(self, target_cable_length, cable_tension_ig, vert_cg):
        def objective(cable_tension_lists):
            vert_force = self.fd_kinodynamics(cable_tension_lists, vert_cg)
            diff_cable_length = np.zeros((self.cds.nCable, 1))
            for i in range(self.cds.nCable):
                idx_pp = self.cds.pp_idx[i]
                diff_cable_length[i] = min(0, target_cable_length[i] - np.linalg.norm(vert_force[idx_pp] - self.cds.pulley_location[i]))
            # print("current cable diff: ", diff_cable_length)
            return np.linalg.norm(diff_cable_length)
        bounds = [(0, 20) for _ in range(self.cds.nCable)]
        initial_guess = cable_tension_ig
        res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, tol = 1e-6)
        cable_tension = res.x
        return cable_tension

    def deform_lengthInput_final(self, target_cable_length, R_list):
        # with the target cable length and rotation matrices, calculate the new position of the pulleys
        # return: selective shape-up-deformation mesh's cur_vertices
        target_tet_sk_list = self.cds.original_tet_sk.copy() + [np.zeros((2, 3)) for _ in range(self.cds.nCable)]
        for i in range(self.cds.nCable):
            unit_vec = self.cds.vertices[self.cds.pp_idx[i]] - self.cds.pulley_location[i]
            unit_vec = unit_vec / np.linalg.norm(unit_vec)
            pp_loc_temp = self.cds.pulley_location[i] + unit_vec * target_cable_length[i]
            cable_array = np.array([self.cds.pulley_location[i], pp_loc_temp])
            target_tet_sk_list[self.cds.num_tet + i] = self.N22 @ cable_array

        cable_considered = [1 for _ in range(self.cds.nCable)]
        
        for i in range(self.cds.nCable):
            matA_this = self.matA_tilde_list[i].copy()
            VecB2Add_this = self.VecB2Add.copy()
            weight_list = self.cds.weight_list.copy() + [self.cds.weight_cable for _ in range(self.cds.nCable)]
            weight_list[self.cds.num_tet + i] = 0
            matAT_this = self.matAT_tilde_list[i]
            matATA_this = self.matATA_tilde_list[i]
            VecB2Add_this[4*self.cds.num_tet + 2*i, :] = 0
            VecB2Add_this[4*self.cds.num_tet + 2*i + 1, :] = 0
            final_cable_length, VectorX_moving, R_list_final = self.deform_once(matAT_this, matATA_this, VecB2Add_this, target_tet_sk_list, weight_list, R_list)
            if final_cable_length[i] < target_cable_length[i]:
                cable_considered[i] = 0

        print("cable_considered: ", cable_considered)

        weight_list = self.cds.weight_list.copy() + [self.cds.weight_cable for _ in range(self.cds.nCable)]
        matA_tilde_this = self.matA_tilde.copy()
        VecB2Add_this = self.VecB2Add.copy()
        for i in range(self.cds.nCable):
            if cable_considered[i] == 0:
                matA_tilde_this[4*self.cds.num_tet + 2*i, :] = 0
                matA_tilde_this[4*self.cds.num_tet + 2*i + 1, :] = 0
                weight_list[self.cds.num_tet + i] = 0
                VecB2Add_this[4*self.cds.num_tet + 2*i, :] = 0
                VecB2Add_this[4*self.cds.num_tet + 2*i + 1, :] = 0

        matAT_tilde = matA_tilde_this.T
        matATA_tilde = matA_tilde_this.T @ matA_tilde_this
        tol_last = 1e-6
        final_cable_length, VectorX_moving, R_list_final = self.deform_once(matAT_tilde, matATA_tilde, VecB2Add_this, target_tet_sk_list, weight_list, R_list, tol_last)
        # update the vertices
        R_list_final = R_list_final[0:self.cds.num_tet]
        cur_vert = self.cds.cur_vertices.copy()
        for i in range(self.cds.nMoving):
            cur_vert[self.cds.idxMoving_2_idxAll[i]] = VectorX_moving[i]
        return cur_vert, final_cable_length, R_list_final
    
    def deform_once(self, matAT, matATA, VecB2Add, target_tet_sk_list, weight_list, R_list, tol=1e-3, max_iter = 1000):
        # LDLT decomposition of ATA
        ts = time.time()
        L, D, perm = ldl(matATA)
        # print("ldlt time: ", time.time() - ts)
        VectorX_moving = np.zeros((self.cds.nMoving, 3))
        VectorX_moving_last = VectorX_moving.copy()
        R_list_inloop = R_list.copy()

        for i in range(max_iter):
            VectorX_moving_last = VectorX_moving.copy()
            VectorBSide = self.construct_vectorBSide(target_tet_sk_list, weight_list, R_list_inloop, VecB2Add)
            Atb = matAT @ VectorBSide
            # solve ATAx = ATb
            for j in range(3):
                b = Atb[:, j]
                Pb = b[perm]
                y = solve_triangular(L, Pb, lower=True)
                z = y / np.diag(D)
                VectorX_moving[:,j] = solve_triangular(L.T, z, lower=False)
            R_list_inloop = self.cal_rotation(VectorX_moving, target_tet_sk_list)
            if (i > 10 and i % 20 == 0):
                diff = np.linalg.norm(VectorX_moving - VectorX_moving_last)
                # print("iter: ", i, "diff: ", diff)
                if diff < tol and i > 10:
                    break
            
        R_list_final = R_list_inloop.copy()
        final_cable_length = self.cds.initial_cable_length.copy()
        for i in range(self.cds.nCable):
            final_cable_length[i] = np.linalg.norm(VectorX_moving[self.cds.idxAll_2_idxMoving[self.cds.pp_idx[i]]] - self.cds.pulley_location[i])
        return final_cable_length, VectorX_moving, R_list_final

    def fd_kinodynamics(self, cable_tension_list, cur_vertices, max_iter=100):
        # with the cable tension list and rotation matrices, calculate the force density vector
        H_mat = np.zeros((3*self.cds.nMoving, self.cds.nCable))
        tol = 1e-6
        q = np.zeros((3*self.cds.nMoving, 1))
        for i in range(self.cds.num_vert):
            idx_moving = self.cds.idxAll_2_idxMoving[i]
            if idx_moving != -1:
                q[3*idx_moving:3*idx_moving+3, 0] = cur_vertices[i]
        q_last = q.copy()
        cable_tensions = np.array(cable_tension_list).reshape((self.cds.nCable, 1))
        grativity_vec_moving = np.zeros((3*self.cds.nMoving, 1))
        for i in range(self.cds.nMoving):
            idx = self.cds.idxMoving_2_idxAll[i]
            grativity_vec_moving[3*i:3*i+3, 0] = self.cds.gravity_vec[3*idx:3*idx+3, 0]
        for i in range(max_iter):
            q_last = q.copy()
            R_list_1212 = self.cal_rotation_fem(q)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_tilde, K_tilde_vec2add = self.assemble_K_tilde(Ke_list)
            f0_tilde = self.assemble_f0_tilde(Ke0_list)
            H_mat = np.zeros((3*self.cds.nMoving, self.cds.nCable))
            for j in range(self.cds.nCable):
                idx_pp_moving = self.cds.idxAll_2_idxMoving[self.cds.pp_idx[j]]
                unit_vec = self.cds.pulley_location[j,:] - q[3*idx_pp_moving:3*idx_pp_moving+3].reshape((3,))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                H_mat[3*idx_pp_moving:3*idx_pp_moving+3, j] = unit_vec
            vecB = H_mat @ cable_tensions + K_tilde_vec2add + f0_tilde + grativity_vec_moving
            # vecB = K_tilde_vec2add + f0_tilde + grativity_vec_moving
            q = np.linalg.solve(K_tilde, vecB)
            diff = np.linalg.norm(q - q_last)
            # print("iter: ", i, "diff: ", diff)
            if diff < tol and i > 2:
                break
        
        cur_vert = self.cds.vertices.copy()
        # update current vertices of cds
        for i in range(self.cds.nMoving):
            cur_vert[self.cds.idxMoving_2_idxAll[i],:] = q[3*i:3*i+3].reshape((3,))
        f_residule = K_tilde @ q - vecB
        return cur_vert
        
    def fd_dynamic(self, cable_tension_lists, time_list, starting_vert, h = 0.1):
        assert len(cable_tension_lists) == len(time_list)
        ntime = len(time_list)
        Q_a = np.zeros((3*self.cds.num_vert, 1))
        for i in range(self.cds.num_vert):
            for j in range(3):
                Q_a[3*i+j, 0] = starting_vert[i][j]
        Q_a_last = np.zeros((3*self.cds.num_vert, 1))
        Q_ad = np.zeros((3*self.cds.num_vert, 1))
        t_a = 0.0
        H_mat = np.zeros((self.cds.nCable, 3*self.cds.num_vert))
        for i in range(ntime):
            cable_tension = np.array(cable_tension_lists[i]).reshape((self.cds.nCable, 1))
            while t_a < time_list[i]:
                Q_a_last = Q_a.copy()
                R_list_1212 = self.cal_rotation_fkLength(Q_a)
                Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
                K_mat = self.assemble_K(Ke_list)
                f0 = self.assemble_f0(Ke0_list)
                A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + h * self.cds.W_mat @ K_mat 
                A_inv = np.linalg.inv(A_mat)
                b_vec = self.cds.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.cds.gravity_vec)
                dv_free = A_inv @ b_vec
                dv_free.reshape((3*self.cds.num_vert, 1))
                Q_free = Q_a + h * Q_ad + h * dv_free
                for j in range(self.cds.nCable):
                    idx_pp = self.cds.pp_idx[j]
                    unit_vec = self.cds.pulley_location[j,:] - Q_free[3*idx_pp:3*idx_pp+3].reshape((3,))
                    unit_vec = unit_vec / np.linalg.norm(unit_vec)
                    # print("unit_vec: ", unit_vec)
                    H_mat[j, 3*idx_pp:3*idx_pp+3] = unit_vec
                dv_cor = A_inv @ (self.cds.W_mat @ H_mat.T @ cable_tension).reshape((3*self.cds.num_vert, 1))
                # print("dv_free norm: ", np.linalg.norm(dv_free))
                dv = dv_free + dv_cor
                Q_ad = Q_ad + dv
                Q_a = Q_a + h * Q_ad
                t_a += h
                diff = np.linalg.norm(Q_a - Q_a_last)
                # print("cur_time: ", t_a, "diff: ", diff)

        vert_force_dynamic = self.cds.cur_vertices.copy()
        for i in range(self.cds.num_vert):
            vert_force_dynamic[i] = Q_a[3*i:3*i+3].reshape((3,))
        return vert_force_dynamic

    def fd_dynamic_once(self, cable_tension_list, starting_vert, h = 0.1):
        Q_a = np.zeros((3*self.cds.num_vert, 1))
        for i in range(self.cds.num_vert):
            for j in range(3):
                Q_a[3*i+j, 0] = starting_vert[i][j]
        Q_a_last = np.zeros((3*self.cds.num_vert, 1))
        Q_ad = np.zeros((3*self.cds.num_vert, 1))
        t_a = 0.0
        H_mat = np.zeros((self.cds.nCable, 3*self.cds.num_vert))
        total_time = 5.0
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + h * self.cds.W_mat @ K_mat 
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.cds.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.cds.gravity_vec)
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.cds.num_vert, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            for j in range(self.cds.nCable):
                idx_pp = self.cds.pp_idx[j]
                unit_vec = self.cds.pulley_location[j,:] - Q_free[3*idx_pp:3*idx_pp+3].reshape((3,))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                H_mat[j, 3*idx_pp:3*idx_pp+3] = unit_vec
            dv_cor = A_inv @ (self.cds.W_mat @ H_mat.T @ cable_tension_list).reshape((3*self.cds.num_vert, 1))
            # print("dv_free norm: ", np.linalg.norm(dv_free))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            diff = np.linalg.norm(Q_a - Q_a_last)/self.cds.num_vert
            if diff < 1e-4:
                break
            print("cur_time: ", t_a, "diff: ", diff)
        vert_force_dynamic = self.cds.cur_vertices.copy()
        for i in range(self.cds.num_vert):
            vert_force_dynamic[i] = Q_a[3*i:3*i+3].reshape((3,))
        return vert_force_dynamic


    def construct_matA(self):
        # construct the matrix A and VecB2Add
        # return the matrixA
        matA = np.zeros((4*self.cds.num_tet + 2*self.cds.nCable, self.cds.num_vert))
        matA_tilde = np.zeros((4*self.cds.num_tet + 2*self.cds.nCable, self.cds.nMoving))
        for i in range(self.cds.num_tet):
            tet = self.cds.tetrahedra[i]
            for j in range(4):
                idxRow = 4*i + j
                for k in range(4):
                    idxCol = tet[k]
                    matA[idxRow, idxCol] = self.N44[j, k] * self.cds.weight_list[i]

        for i in range(self.cds.nCable):
            for j in range(2):
                idxRow = 4*self.cds.num_tet + 2*i + j
                idxCol = self.cds.pp_idx[i]
                matA[idxRow, idxCol] = self.N22[j, 1] * self.cds.weight_cable
        
        VecB2Add = np.zeros((4*self.cds.num_tet + 2*self.cds.nCable, 3))

        for i in range(self.cds.num_tet):
            for j in range(self.cds.num_vert):
                idx_moving = self.cds.idxAll_2_idxMoving[j]
                if idx_moving != -1:
                    for k in range(4):
                        idxRow = 4*i + k
                        matA_tilde[idxRow, idx_moving] = matA[idxRow, j]
                else:
                    for k in range(4):
                        idxRow = 4*i + k
                        VecB2Add[idxRow, :] -= matA[idxRow, j] * self.cds.vertices[j]

        for i in range(self.cds.nCable):
            idxRow = 4*self.cds.num_tet + 2*i
            idx_moving = self.cds.idxAll_2_idxMoving[self.cds.pp_idx[i]]
            matA_tilde[idxRow, idx_moving] = matA[idxRow, self.cds.pp_idx[i]]
            matA_tilde[idxRow+1, idx_moving] = matA[idxRow+1, self.cds.pp_idx[i]]
            VecB2Add[4*self.cds.num_tet + 2*i, :] -= 1/2 * self.cds.pulley_location[i] * self.cds.weight_cable
            VecB2Add[4*self.cds.num_tet + 2*i + 1, :] += 1/2 * self.cds.pulley_location[i] * self.cds.weight_cable

        matA_tilde_list = [matA_tilde.copy() for _ in range(self.cds.nCable)]
        for i in range(self.cds.nCable):
            matA_tilde_list[i][4*self.cds.num_tet + 2*i, :] = 0
            matA_tilde_list[i][4*self.cds.num_tet + 2*i + 1, :] = 0

        matAT_tilde_list = [matA_tilde.T for matA_tilde in matA_tilde_list]
        matATA_tilde_list = [matA_tilde.T @ matA_tilde for matA_tilde in matA_tilde_list]

        return matA_tilde, matA_tilde_list, matAT_tilde_list, matATA_tilde_list, VecB2Add

    def construct_vectorBSide(self, target_tet_sk_list, weight_list, R_list, VecB2Add):
        # construct the b side of the linear system
        # return the vectorBSide
        vectorBSide = np.zeros((4*self.cds.num_tet + 2*self.cds.nCable, 3))
        for i in range(self.cds.num_tet):
            target_tet_sk = target_tet_sk_list[i]
            R_this = R_list[i]
            vectorBSide[4*i:4*i+4, :] = (R_this @ target_tet_sk.T).T * weight_list[i]

        for i in range(self.cds.nCable):
            target_cable_sk = target_tet_sk_list[self.cds.num_tet + i]
            R_this = R_list[self.cds.num_tet + i]
            vectorBSide[4*self.cds.num_tet + 2*i:4*self.cds.num_tet + 2*i+2, :] = (R_this @ target_cable_sk.T).T * weight_list[self.cds.num_tet + i]
            
        vectorBSide += VecB2Add
        return vectorBSide

    def cal_rotation(self, VectorXMoving, target_tet_sk_list):
        # local step to update the vertices
        R_list = [np.zeros((3, 3)) for _ in range(self.cds.num_tet + self.cds.nCable)]
        
        for i in range(self.cds.num_tet):
            # use svd to calculate the rotation matrix
            cur_tet = np.zeros((4, 3))
            for j in range(4):
                idx = self.cds.tetrahedra[i][j]
                if self.cds.idxAll_2_idxMoving[idx] != -1:
                    cur_tet[j, :] = VectorXMoving[self.cds.idxAll_2_idxMoving[idx], :]
                else:
                    cur_tet[j, :] = self.cds.vertices[idx]
            cur_tet_sk = self.N44 @ cur_tet
            u, s, vh = np.linalg.svd(cur_tet_sk.T @ target_tet_sk_list[i])
            R_list[i] = u @ vh

        for i in range(self.cds.nCable):
            # use svd to calculate the rotation matrix
            cur_cable = np.zeros((2, 3))
            cur_cable[0, :] = self.cds.pulley_location[i]
            cur_cable[1, :] = VectorXMoving[self.cds.idxAll_2_idxMoving[self.cds.pp_idx[i]], :]
            cur_cable_sk = self.N22 @ cur_cable
            u, s, vh = np.linalg.svd(cur_cable_sk.T @ target_tet_sk_list[self.cds.num_tet + i])
            R_list[self.cds.num_tet + i] = u @ vh
        return R_list.copy()

    def cal_rotation_fkLength(self, Q):
        R_list = [np.zeros((12, 12)) for _ in range(self.cds.num_tet + self.cds.nCable)]
        for i in range(self.cds.num_tet):
            cur_tet = np.zeros((4, 3))
            for j in range(4):
                idx = self.cds.tetrahedra[i][j]
                cur_tet[j, :] = Q[3*idx:3*idx+3].reshape((3,))
            cur_tet_sk = self.N44 @ cur_tet
            u, s, vh = np.linalg.svd(cur_tet_sk.T @ self.cds.original_tet_sk[i])
            R_this = u @ vh
            for j in range(4):
                R_list[i][3*j:3*j+3, 3*j:3*j+3] = R_this
        return R_list

    def cal_rotation_femik(self, q_moving, target_ee_pos, idxAll_2_idxMoving_ik):
        R_list_1212 = [np.eye(12) for _ in range(self.cds.num_tet)]
        for i in range(self.cds.num_tet):
            tet = self.cds.tetrahedra[i]
            cur_tet = np.zeros((4, 3))
            for j in range(4):
                idx = tet[j]
                idx_moving = idxAll_2_idxMoving_ik[idx]
                if idx_moving != -1 and idx_moving != -2:
                    cur_tet[j, :] = q_moving[3*idx_moving:3*idx_moving+3].reshape((3,))
                else:
                    if idx_moving == -1:
                        cur_tet[j, :] = self.cds.vertices[idx]
                    elif idx_moving == -2:
                        idxee = self.cds.ee_idxs.index(idx)
                        cur_tet[j, :] = target_ee_pos[idxee, :]

            cur_tet_sk = self.N44 @ cur_tet
            u, s, vh = np.linalg.svd(cur_tet_sk.T @ self.cds.original_tet_sk[i])
            R_this = u @ vh
            for j in range(4):
                R_list_1212[i][3*j:3*j+3, 3*j:3*j+3] = R_this
        return R_list_1212

    def generate_ws(self, ws_file, nc_each = 10):
        init_cable_length = self.cds.initial_cable_length.copy()
        shorten_ratio = 0.3
        min_cable_length = [c*shorten_ratio for c in init_cable_length]
        total_nws = nc_each ** self.cds.nCable
        ws_list_length = [[0 for _ in range(self.cds.nCable)] for _ in range(total_nws)]
        ws_list_ee = [np.zeros((self.cds.n_ee, 3)) for _ in range(total_nws)]
        ws_list_force = [np.zeros((self.cds.nCable, 1)) for _ in range(total_nws)]
        # open the file, append at each line
        R_initial = [np.eye(3) for _ in range(self.cds.num_tet + self.cds.nCable)]
        f = open(ws_file, "a")
        start_vert = self.cds.vertices.copy()
        vert_length = start_vert.copy()
        # start_vert = start_vert.reshape((3*self.cds.num_vert, 1))
        for i in range(total_nws):
            if i == 0:
                continue
            start_vert = vert_length
            ws_list_this = [0 for _ in range(self.cds.nCable)]
            cable_length_list = [0.0 for _ in range(self.cds.nCable)]
            for j in range(self.cds.nCable):
                ws_list_this[j] = i % nc_each
                i = i // nc_each
            for j in range(self.cds.nCable):
                cable_length_list[j] = init_cable_length[j] - ws_list_this[j] * (init_cable_length[j] - min_cable_length[j]) / (nc_each - 1)
            print("generating ws for cable length: ", cable_length_list)
            vert_length, cable_tension_final, R_list_1212 = self.fk_length_startMesh(cable_length_list, start_vert)
            pp_loc = np.zeros((self.cds.nCable, 3))
            for j in range(self.cds.nCable):
                pp_loc[j, :] = vert_length[self.cds.pp_idx[j]]
            is_intersect, intesect_array = self.check_intersection(pp_loc, vert_length)
            cable_tension_final = cable_tension_final.tolist()
            cable_length_final = [0 for _ in range(self.cds.nCable)]
            for j in range(self.cds.nCable):
                cable_length_final[j] = round(np.linalg.norm(vert_length[self.cds.pp_idx[j]] - self.cds.pulley_location[j]),4) # 0.1 mm
            ws_list_length[i] = np.array(cable_length_final)
            for j in range(self.cds.nCable):
                ws_list_force[i][j] = round(cable_tension_final[j],4) # 0.1 mm
            for j in range(self.cds.n_ee):
                ws_list_ee[i][j] = vert_length[self.cds.ee_idxs[j]]
            str_2Write = str(ws_list_length[i]) + ", " + str(ws_list_force[i]) + ", "
            for j in range(self.cds.n_ee):
                str_2Write += str(ws_list_ee[i][j]) + ", "
            f.write(str_2Write + "\n")

    # IK part
    def ik_length_cg(self, ee_target_position, weight_ee_list, cable_length_initial, R_list_initial):
        # with the end-effector target location, calculate the cable length
        # return: the cable length
        def objective(cable_length):
            vert_cg, cable_length_now, R_list_now = self.fk_length_allCableTensioned(cable_length, R_list_initial)
            diff = 0
            for i in range(self.cds.n_ee):
                diff += weight_ee_list[i] * np.linalg.norm(vert_cg[self.cds.ee_idxs[i]] - ee_target_position[i])
            print("cur cable length: ", cable_length, "diff: ", diff)
            return diff
        bound_ratio = 0.2
        bounds = [(self.cds.initial_cable_length[i]*bound_ratio, self.cds.initial_cable_length[i]) for i in range(self.cds.nCable)]
        initial_guess = cable_length_initial
        res = minimize(objective, x0 = initial_guess, method='SLSQP', bounds=bounds, tol = 1e-3, options={'eps': 1e-3})
        print("res: ", res)
        cable_lengh_final = res.x
        return cable_lengh_final

    def ik_force_integrate(self, ee_target_position, weight_ee_list, initial_cable_tension, initial_vert, initial_R_list_1212):
        def objective(cable_tension_list):
            cable_tension_list = cable_tension_list.reshape((self.cds.nCable, 1))
            tol = 1e-6
            total_time = 5.0
            h = 0.01
            t_a = 0.0
            R_list_1212 = initial_R_list_1212
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            Q_a = np.zeros((3*self.cds.num_vert, 1))
            Q_a_last = np.zeros((3*self.cds.num_vert, 1))
            Q_ad = np.zeros((3*self.cds.num_vert, 1))
            for i in range(self.cds.num_vert):
                for j in range(3):
                    Q_a[3*i+j, 0] = initial_vert[i][j]
            total_time = 5.0
            h = 0.01
            t_a = 0.0
            A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + h * self.cds.W_mat @ K_mat 
            A_inv = np.linalg.inv(A_mat)
            
            while t_a < total_time:
                Q_a_last = Q_a.copy()
                H_mat = np.zeros((self.cds.nCable, 3*self.cds.num_vert))
                b_vec = self.cds.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.cds.gravity_vec)
                dv_free = (A_inv @ b_vec).reshape((3*self.cds.num_vert, 1))
                Q_free = Q_a + h * Q_ad + h * dv_free
                for i in range(self.cds.nCable):
                    idx_pp = self.cds.pp_idx[i]
                    unit_vec = (self.cds.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                    unit_vec = unit_vec / np.linalg.norm(unit_vec)
                    # print("unit_vec: ", unit_vec)
                    H_mat[i, 3*idx_pp:3*idx_pp+3] = unit_vec
                dv_cor = A_inv @ (self.cds.W_mat @ H_mat.T @ cable_tension_list).reshape((3*self.cds.num_vert, 1))
                Q_ad = Q_ad + dv_free + dv_cor
                Q_a = Q_a + h * Q_ad
                R_list_1212 = self.cal_rotation_fkLength(Q_a)
                Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
                K_mat = self.assemble_K(Ke_list)
                f0 = self.assemble_f0(Ke0_list)
                A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + h * self.cds.W_mat @ K_mat
                A_inv = np.linalg.inv(A_mat)
                t_a += h
                if np.linalg.norm(Q_a - Q_a_last) < tol or np.linalg.norm(Q_ad) < tol:
                    break
            
            diff_ee = 0
            for i in range(self.cds.n_ee):
                diff_ee += weight_ee_list[i] * np.linalg.norm(Q_a[3*self.cds.ee_idxs[i]:3*self.cds.ee_idxs[i]+3].reshape((3,)) - ee_target_position[i])
            print("cable_tension: ", cable_tension_list, "diff_ee: ", diff_ee)
            return diff_ee
        
        bounds = [(0, 120) for _ in range(self.cds.nCable)]
        initial_guess = initial_cable_tension.copy()
        res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, tol = 1e-6)
        cable_tension_list = res.x
        return cable_tension_list

    def ik_force_kineto(self, ee_target_position, weight_ee_list, initial_cable_tension, initial_vert):
        def objective(cable_tension_list):
            cable_tension_list = cable_tension_list.reshape((self.cds.nCable, 1))
            vert_force = self.fd_kinodynamics(cable_tension_list, self.cds.vertices)
            diff_ee = 0
            for i in range(self.cds.n_ee):
                diff_ee += weight_ee_list[i] * np.linalg.norm(vert_force[self.cds.ee_idxs[i]] - ee_target_position[i])
            # print("cable_tension: ", cable_tension_list, "diff_ee: ", diff_ee)
            print("diff_ee: ", diff_ee)
            return diff_ee
        
        bounds = [(0, 120) for _ in range(self.cds.nCable)]
        initial_guess = initial_cable_tension.copy()
        res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, tol = 1e-8, options={'eps': 1e-2})
        cable_tension_list = res.x
        return cable_tension_list

    def fk_length_printws(self, target_cable_length):
        vert_cg, vert_length, cable_tension_ig = self.deform_lengthInput_final(target_cable_length, [np.eye(3) for _ in range(self.cds.num_tet + self.cds.nCable)])
        ws_pos = np.array([vert_cg[self.cds.ee_idxs[i]] for i in range(self.cds.n_ee)])
        print("cable length: ", target_cable_length)
        print("ws_pos: ", ws_pos)

    def cal_ik_ig(self, ee_target_position): # force the end-effector to the target position, calculate its final config
        idxAll_2_idxMoving_ik = self.cds.idxAll_2_idxMoving.copy()
        n_moving = 0
        for i in range(self.cds.num_vert):
            if i in self.cds.fixed_idx:
                idxAll_2_idxMoving_ik[i] = -1
            elif i in self.cds.ee_idxs:
                idxAll_2_idxMoving_ik[i] = -2
            else:
                idxAll_2_idxMoving_ik[i] = n_moving
                n_moving += 1
        nMoving_ik = self.cds.nMoving - self.cds.n_ee
        # print("nMoving_ik: ", nMoving_ik)
        grativity_vec_moving = np.zeros((3*nMoving_ik, 1))
        for i in range(self.cds.num_vert):
            idx = idxAll_2_idxMoving_ik[i]
            if idx != -1 and idx != -2:
                grativity_vec_moving[3*idx:3*idx+3, 0] = self.cds.gravity_vec[3*i:3*i+3, 0]
        q = np.zeros((3*nMoving_ik, 1)) # initial guess
        for i in range(self.cds.num_vert):
            idx = idxAll_2_idxMoving_ik[i]
            if idx != -1 and idx != -2:
                q[3*idx:3*idx+3, 0] = self.cds.vertices[i]
        q_last = q.copy()
        max_iter = 1000
        tol = 1e-8
        for num_iter in range(max_iter):
            q_last = q.copy()
            R_list_1212 = self.cal_rotation_femik(q, ee_target_position, idxAll_2_idxMoving_ik)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_tilde, K_tilde_vec2add = self.assemble_K_ik(Ke_list, ee_target_position, idxAll_2_idxMoving_ik)
            f0_tilde = self.assemble_f0_ik(Ke0_list, idxAll_2_idxMoving_ik)
            b_vec = K_tilde_vec2add + f0_tilde + grativity_vec_moving
            q = np.linalg.solve(K_tilde, b_vec)
            diff = np.linalg.norm(q - q_last)
            # print("iter: ", num_iter, "diff: ", diff)
            if diff < tol and num_iter > 2:
                break
        cur_vert = self.cds.vertices.copy()
        for i in range(self.cds.n_ee):
            cur_vert[self.cds.ee_idxs[i]] = ee_target_position[i]
        for i in range(self.cds.num_vert):
            idxMovingik = idxAll_2_idxMoving_ik[i]
            if idxMovingik != -1 and idxMovingik != -2:
                cur_vert[i] = q[3*idxMovingik:3*idxMovingik+3].reshape((3,))
        final_cable_length = self.cds.initial_cable_length.copy()
        for i in range(self.cds.nCable):
            idx_pp = self.cds.pp_idx[i]
            final_cable_length[i] = np.linalg.norm(cur_vert[idx_pp] - self.cds.pulley_location[i])
        return cur_vert, final_cable_length
    
    def ik_trajectory_kineto(self, ee_target_position_list, weight_ee_list, initial_cable_tension, initial_vert):
        final_cable_tension_list = []
        final_cable_length_list = []
        vert_lists = []
        n_points = len(ee_target_position_list)
        cable_length_list = self.cds.initial_cable_length.copy()
        for i in range(n_points):
            ee_target_position = ee_target_position_list[i]
            weight_ee = weight_ee_list[i]
            cable_tension_list = self.ik_force_kineto(ee_target_position, weight_ee, initial_cable_tension, initial_vert)
            vert_force = self.fd_kinodynamics(cable_tension_list, initial_vert)
            for j in range(self.cds.nCable):
                cable_length_list[j] = round(np.linalg.norm(vert_force[self.cds.pp_idx[j]] - self.cds.pulley_location[j]), 4) # 0.1 mm
                cable_tension_list[j] = round(cable_tension_list[j], 2) # 0.01N
            final_cable_tension_list.append(cable_tension_list)
            final_cable_length_list.append(cable_length_list.copy())
            vert_lists.append(vert_force.copy())
            initial_cable_tension = cable_tension_list.copy()

        return final_cable_tension_list, final_cable_length_list, vert_lists

    def ik_trajectory_cg(self, ee_target_position_list, weight_ee_list):
        final_cable_tension_list = []
        vert_lists = []
        n_points = len(ee_target_position_list)
        cable_tension_list = [0.0 for _ in range(self.cds.nCable)]
        for i in range(n_points):
            ee_target_position = ee_target_position_list[i]
            weight_ee = weight_ee_list[i]
            cur_vert, cable_length_ig = self.cal_ik_ig(ee_target_position)
            vert_length, cable_tension, R_list_1212 = self.fk_length_allCableTensioned(cable_length_ig, [np.eye(3) for _ in range(self.cds.num_tet + self.cds.nCable)])
            cable_tension_list_final = self.ik_force_kineto(ee_target_position, weight_ee, cable_tension, vert_length)
            
            for j in range(self.cds.nCable):
                cable_tension_list_final[j] = round(cable_tension_list_final[j], 2)
            final_cable_tension_list.append(cable_tension_list_final.copy())
            print("cable_tension: ", cable_tension_list_final)
        return final_cable_tension_list

    def fk_length_forJac(self, target_cable_length, starting_q, tol = 1e-4):
        # with the end-effector target location, calculate the cable length
        # return: the cable length
        Q_a = starting_q.copy()
        Q_ad = np.zeros((3*self.cds.num_vert, 1))
        Q_a_last = Q_a.copy()
        total_time = 5.0
        h = 0.01
        t_a = 0.0
        phi_Qfree = np.zeros((self.cds.nCable, 1))
        H_free = np.zeros((self.cds.nCable, 3*self.cds.num_vert))
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + h * self.cds.W_mat @ K_mat 
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.cds.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.cds.gravity_vec)
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.cds.num_vert, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            for i in range(self.cds.nCable):
                idx_pp = self.cds.pp_idx[i]
                unit_vec = (self.cds.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                phi_Qfree[i] = target_cable_length[i] - np.linalg.norm(self.cds.pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
            lcp_Mmat = h * H_free @ self.cds.W_mat @ A_inv @ H_free.T
            # check if lcp_Mmat is PD
            M_is_PD = np.all(np.linalg.eigvals(lcp_Mmat) > 0)
            lcp_q = phi_Qfree.reshape((self.cds.nCable,))
            lcp_sol = qe.optimize.lcp_lemke(lcp_Mmat, lcp_q)
            if not lcp_sol.success:
                print("lcp failed: ")
                break
            cable_tension = lcp_sol.z
            # print("cur time: ", t_a, "cable_tension: ", cable_tension, "lcp_q: ", lcp_q.flatten(), "M_is_PD: ", M_is_PD)
            # cable_tension.reshape((self.cds.nCable, 1))
            dv_cor = A_inv @ (self.cds.W_mat @ H_free.T @ cable_tension).reshape((3*self.cds.num_vert, 1))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            if np.linalg.norm(Q_a - Q_a_last)/self.cds.num_vert < tol: # average movement of each vertex is smaller than 0.1mm
                break
        return Q_a

    def ik_getJacobian_force(self, ee_target_position, weight_ee_list, start_q, start_cableTension):
        pass

    def ik_getJacobian_length(self, ee_target_position, weight_ee_list, start_q, start_cableLength, dl = 1e-3, tol_fk = 1e-5): # 1mm
        diff_list_jac = np.zeros((1, self.cds.nCable))
        for i in range(self.cds.nCable):
            tar_cable_length_minus = start_cableLength.copy()
            tar_cable_length_plus = start_cableLength.copy()
            tar_cable_length_minus[i] -= dl
            tar_cable_length_plus[i] += dl
            q_minus = self.fk_length_forJac(tar_cable_length_minus, start_q, tol_fk)
            q_plus = self.fk_length_forJac(tar_cable_length_plus, start_q, tol_fk)
            cur_ee_pos_minus = np.zeros((self.cds.n_ee, 3))
            cur_ee_pos_plus = np.zeros((self.cds.n_ee, 3))
            cur_diff_minus = 0.0
            cur_diff_plus = 0.0
            for j in range(self.cds.n_ee):
                cur_ee_pos_minus[j, :] = q_minus[3*self.cds.ee_idxs[j]: 3*self.cds.ee_idxs[j]+3].reshape((3,))
                cur_ee_pos_plus[j, :] = q_plus[3*self.cds.ee_idxs[j]: 3*self.cds.ee_idxs[j]+3].reshape((3,))
                cur_diff_minus += weight_ee_list[j] * np.linalg.norm(cur_ee_pos_minus[j, :] - ee_target_position[j,:])
                cur_diff_plus += weight_ee_list[j] * np.linalg.norm(cur_ee_pos_plus[j, :] - ee_target_position[j,:])
            diff_list_jac[0, i] = (cur_diff_plus - cur_diff_minus) / (2 * dl)
        return diff_list_jac

    def jac_based_ik_length(self, ee_target_position, weight_ee_list, start_q):
        diff_ee_tol = 1e-3 # 0.5mm
        max_iter = 100 
        dl = 5e-4
        start_ee = np.zeros((self.cds.n_ee, 3))
        start_cable_length = self.cds.initial_cable_length.copy()
        cur_diff = 0.0
        for i in range(self.cds.n_ee):
            start_ee[i, :] = start_q[3*self.cds.ee_idxs[i]: 3*self.cds.ee_idxs[i]+3].reshape((3,))
            start_cable_length[i] = np.linalg.norm(start_ee[i, :] - self.cds.pulley_location[i])
            cur_diff += weight_ee_list[i] * np.linalg.norm(start_ee[i, :] - ee_target_position[i,:])
        num_iter = 0
        shrinking_rate = 0.2
        while cur_diff > diff_ee_tol and num_iter < max_iter:
            if cur_diff < 1e-3:
                tol_fk = 1e-5
                dl = 5e-5
                diff_list_jac = self.ik_getJacobian_length(ee_target_position, weight_ee_list, start_q, start_cable_length, dl, tol_fk)
            else:
                tol_fk = 1e-4
                dl = 5e-4
                diff_list_jac = self.ik_getJacobian_length(ee_target_position, weight_ee_list, start_q, start_cable_length)
            tau_min = 1e3
            for i in range(self.cds.nCable):
                temp = np.abs(cur_diff) / np.abs(diff_list_jac[0, i])
                if temp < tau_min:
                    tau_min = temp
            new_diff = cur_diff + 1.0
            tau_2test = tau_min 
            # shrinking process
            shrink_num = 0
            diff_last = cur_diff
            while 1:
                cable_length_cmd = start_cable_length.copy()
                for i in range(self.cds.nCable):
                    cable_length_cmd[i] -= tau_2test * diff_list_jac[0, i]
                q_new = self.fk_length_forJac(cable_length_cmd, start_q, tol_fk)
                new_ee_pos = np.zeros((self.cds.n_ee, 3))
                new_diff = 0.0
                for i in range(self.cds.n_ee):
                    new_ee_pos[i, :] = q_new[3*self.cds.ee_idxs[i]: 3*self.cds.ee_idxs[i]+3].reshape((3,))
                    new_diff += weight_ee_list[i] * np.linalg.norm(new_ee_pos[i, :] - ee_target_position[i,:])
                if new_diff < cur_diff:
                    break
                if shrink_num > 0 and new_diff > diff_last:
                    tau_2test /= shrinking_rate
                    break
                diff_last = new_diff
                tau_2test *= shrinking_rate
                shrink_num += 1
                # print("shrink number: ", shrink_num, "new_diff: ", new_diff, ". tau_2test: ", tau_2test)
            # print("shrink process for iter: ", num_iter, "shrink_num: ", shrink_num)
            tau_step = tau_2test
            tau_final = tau_2test
            q_last = q_new.copy()
            cable_length_last = cable_length_cmd.copy()
            for i in range(self.cds.nCable):
                cable_length_last[i] = np.linalg.norm(q_last[3*self.cds.pp_idx[i]: 3*self.cds.pp_idx[i]+3].reshape((3,)) - self.cds.pulley_location[i])
            last_diff = new_diff
            # line search process
            line_search_num = 0
            while 1:
                last_diff = new_diff
                q_last = q_new.copy()
                tau_2test += tau_step
                cable_length_cmd = start_cable_length.copy()
                for i in range(self.cds.nCable):
                    cable_length_cmd[i] -= tau_2test * diff_list_jac[0, i]
                q_new = self.fk_length_forJac(cable_length_cmd, start_q)
                new_ee_pos = np.zeros((self.cds.n_ee, 3))
                new_diff = 0.0
                for i in range(self.cds.n_ee):
                    new_ee_pos[i, :] = q_new[3*self.cds.ee_idxs[i]: 3*self.cds.ee_idxs[i]+3].reshape((3,))
                    new_diff += weight_ee_list[i] * np.linalg.norm(new_ee_pos[i, :] - ee_target_position[i,:])
                if new_diff > cur_diff:
                    tau_final = tau_2test - tau_step
                    for i in range(self.cds.nCable):
                        cable_length_last[i] = np.linalg.norm(q_last[3*self.cds.pp_idx[i]: 3*self.cds.pp_idx[i]+3].reshape((3,)) - self.cds.pulley_location[i])
                    break
                last_diff = new_diff
                q_last = q_new.copy()
                line_search_num += 1
                # print("line_search_num: ", line_search_num, ". new_diff: ", new_diff)
            start_cable_length = cable_length_last.copy()
            start_q = q_last.copy()
            cur_diff = last_diff
            num_iter += 1
            print("iter: ", num_iter, "cur_diff: ", cur_diff, "cable_length diff: ", [a-b for (a,b) in zip(self.cds.initial_cable_length,start_cable_length)], "tau_final: ", tau_final)
        return start_cable_length

    # FEM part
    def assemble_K(self, Ke_list):
        K = np.zeros((3*self.cds.num_vert, 3*self.cds.num_vert))
        for i in range(self.cds.num_tet):
            tet = self.cds.tetrahedra[i]
            for j in range(4):
                idxj = tet[j]
                for k in range(4):
                    idxk = tet[k]
                    K[3*idxj:3*idxj+3, 3*idxk:3*idxk+3] += Ke_list[i][3*j:3*j+3, 3*k:3*k+3]
        return K

    def assemble_K_tilde(self, Ke_list):
        K_tilde = np.zeros((3*self.cds.nMoving, 3*self.cds.nMoving))
        K_tilde_vec2add = np.zeros((3*self.cds.nMoving, 1))
        for i in range(self.cds.num_tet):
            for j in range(4):
                idx_row = self.cds.tetrahedra[i][j]
                idx_moving_row = self.cds.idxAll_2_idxMoving[idx_row]
                if idx_moving_row != -1:
                    for k in range(4):
                        idx_col = self.cds.tetrahedra[i][k]
                        idx_moving_col = self.cds.idxAll_2_idxMoving[idx_col]
                        if idx_moving_col != -1:
                            K_tilde[3*idx_moving_row:3*idx_moving_row+3, 3*idx_moving_col:3*idx_moving_col+3] += Ke_list[i][3*j:3*j+3, 3*k:3*k+3]
                        else:
                            K_tilde_vec2add[3*idx_moving_row:3*idx_moving_row+3, 0] -= Ke_list[i][3*j:3*j+3, 3*k:3*k+3] @ self.cds.vertices[idx_col]
        return K_tilde, K_tilde_vec2add

    def assemble_K_ik(self, Ke_list, ee_pos, idxAll_2_idxMoving_ik):
        K_tilde = np.zeros((3*(self.cds.nMoving - self.cds.n_ee), 3*(self.cds.nMoving - self.cds.n_ee)))
        K_tilde_vec2add = np.zeros((3*(self.cds.nMoving - self.cds.n_ee), 1))
        for i in range(self.cds.num_tet):
            for j in range(4):
                idx_row = self.cds.tetrahedra[i][j]
                idx_moving_row = idxAll_2_idxMoving_ik[idx_row]
                if idx_moving_row != -1 and idx_moving_row != -2:
                    for k in range(4):
                        idx_col = self.cds.tetrahedra[i][k]
                        idx_moving_col = idxAll_2_idxMoving_ik[idx_col]
                        if idx_moving_col != -1 and idx_moving_col != -2:
                            K_tilde[3*idx_moving_row:3*idx_moving_row+3, 3*idx_moving_col:3*idx_moving_col+3] += Ke_list[i][3*j:3*j+3, 3*k:3*k+3]
                        else:
                            if idx_moving_col == -1:
                                K_tilde_vec2add[3*idx_moving_row:3*idx_moving_row+3, 0] -= Ke_list[i][3*j:3*j+3, 3*k:3*k+3] @ self.cds.vertices[idx_col]
                            elif idx_moving_col == -2:
                                K_tilde_vec2add[3*idx_moving_row:3*idx_moving_row+3, 0] -= Ke_list[i][3*j:3*j+3, 3*k:3*k+3] @ ee_pos[self.cds.ee_idxs.index(idx_col)]
        return K_tilde, K_tilde_vec2add

    def assemble_f0(self, Ke0_list):
        f0 = np.zeros((3*self.cds.num_vert, 1))
        q0e = np.zeros((12, 1))
        for i in range(self.cds.num_tet):
            q0e = np.zeros((12, 1))
            for j in range(4):
                idx = self.cds.tetrahedra[i][j]
                q0e[3*j:3*j+3, 0] = self.cds.vertices[idx,:]
            f0e = Ke0_list[i] @ q0e
            for j in range(4):
                idx = self.cds.tetrahedra[i][j]
                f0[3*idx:3*idx+3, 0] += f0e[3*j:3*j+3, 0]
        return f0
    
    def assemble_f0_tilde(self, Ke0_list):
        f0_tilde = np.zeros((3*self.cds.nMoving, 1))
        q0e = np.zeros((12, 1))
        for i in range(self.cds.num_tet):
            q0e = np.zeros((12, 1))
            for j in range(4):
                idx = self.cds.tetrahedra[i][j]
                q0e[3*j:3*j+3, 0] = self.cds.vertices[idx,:]
            f0e = Ke0_list[i] @ q0e
            for j in range(4):
                idx = self.cds.tetrahedra[i][j]
                idx_moving = self.cds.idxAll_2_idxMoving[idx]
                if idx_moving != -1:
                    f0_tilde[3*idx_moving:3*idx_moving+3, 0] += f0e[3*j:3*j+3, 0]
        return f0_tilde

    def assemble_f0_ik(self, Ke0_list, idxAll_2_idxMoving_ik):
        f0_tilde = np.zeros((3*(self.cds.nMoving - self.cds.n_ee), 1))
        q0e = np.zeros((12, 1))
        for i in range(self.cds.num_tet):
            q0e = np.zeros((12, 1))
            for j in range(4):
                idx = self.cds.tetrahedra[i][j]
                q0e[3*j:3*j+3, 0] = self.cds.vertices[idx,:]
            f0e = Ke0_list[i] @ q0e
            for j in range(4):
                idx = self.cds.tetrahedra[i][j]
                idx_moving = idxAll_2_idxMoving_ik[idx]
                if idx_moving != -1 and idx_moving != -2:
                    f0_tilde[3*idx_moving:3*idx_moving+3, 0] += f0e[3*j:3*j+3, 0]
        return f0_tilde

    def cal_Ke_lists(self, R_list_1212):
        Ke_list = [np.zeros((12, 12)) for _ in range(self.cds.num_tet)]
        Ke0_list = [np.zeros((12, 12)) for _ in range(self.cds.num_tet)]
        for i in range(self.cds.num_tet):
            Ke_list[i] = R_list_1212[i] @ self.cds.stiffness_matrix_list[i] @ R_list_1212[i].T @ self.N1212
            Ke0_list[i] = R_list_1212[i] @ self.cds.stiffness_matrix_list[i] @ self.N1212
        return Ke_list, Ke0_list

    def cal_rotation_fem(self, q_moving):
        R_list_1212 = [np.eye(12) for _ in range(self.cds.num_tet)]
        for i in range(self.cds.num_tet):
            tet = self.cds.tetrahedra[i]
            cur_tet = np.zeros((4, 3))
            for j in range(4):
                idx = tet[j]
                idx_moving = self.cds.idxAll_2_idxMoving[idx]
                if idx_moving != -1:
                    cur_tet[j, :] = q_moving[3*idx_moving:3*idx_moving+3].reshape((3,))
                else:
                    cur_tet[j, :] = self.cds.vertices[idx,:]
            cur_tet_sk = self.N44 @ cur_tet
            u, s, vh = np.linalg.svd(cur_tet_sk.T @ self.cds.original_tet_sk[i])
            R_this = u @ vh
            for j in range(4):
                R_list_1212[i][3*j:3*j+3, 3*j:3*j+3] = R_this
        return R_list_1212

    def lemkelcp(self, M,q,maxIter=100):
        """
        sol = lemkelcp(M,q,maxIter)

        Uses Lemke's algorithm to copute a solution to the 
        linear complementarity problem:

        Mz + q >= 0
        z >= 0 
        z'(Mz+q) = 0

        The inputs are given by:

        M - an nxn numpy array
        q - a length n numpy array 
        maxIter - an optional number of pivot iterations. Set to 100 by default

        The solution is a tuple of the form:

        z,exit_code,exit_string = sol

        The entries are summaries in the table below:
        
        |z                | exit_code | exit_string               | 
        -----------------------------------------------------------
        | solution to LCP |    0      | 'Solution Found'          |
        | None            |    1      | 'Secondary ray found'     |
        | None            |    2      | 'Max Iterations Exceeded' |    
        """

        tableau = lemketableau(M,q,maxIter)
        return tableau.lemkeAlgorithm()

    # contact modeling part
    def generate_onews_length(self, cable_command, ws_file, nContact):
        vert_length, cable_tension_noContact, R_list = self.fk_length_startMesh(cable_command, self.cds.vertices)
        n_contactVert = len(self.cds.contact_vertices)
        contactConfig = np.zeros((n_contactVert, 3))
        for i in range(n_contactVert):
            for j in range(3): # round to 0.1 mm
                contactConfig[i, j] = round(vert_length[self.cds.contact_vertices[i], j], 4)
        cl_return = [0.0 for _ in range(self.cds.nCable)]
        for i in range(self.cds.nCable):
            idx_pp = self.cds.pp_idx[i]
            cl_return[i] = round(np.linalg.norm(vert_length[idx_pp] - self.cds.pulley_location[i]),4)
        print("command cable length: ", cable_command, "returned cable length: ", cl_return)
        cable_tension_rounded = [round(c, 4) for c in cable_tension_noContact]
        # append the result to the file, one line of cl_return, one line of cable_tension_rounded, then contactConfig one row by one row
        f = open(ws_file, "a")
        f.write(str(cl_return) + "\n")
        f.write(str(cable_tension_rounded) + "\n")
        for i in range(nContact):
            f.write(str(contactConfig[i]) + "\n")
        f.close()   
        print("generated ws for cable length: ", cl_return)
        return cl_return, cable_tension_rounded, contactConfig

    def generate_contact_ws(self, ws_file):
        init_cable_length = self.cds.initial_cable_length.copy()
        shorten_dist_max = 0.1 # maximum shorten distance is 10 cm
        stepSize = 0.003 # every 1mm
        cl1_2test = init_cable_length[1]
        nContact = len(self.cds.contact_vertices)
        # open the file, write cds pulley location, cds pullpos
        f = open(ws_file, "w")
        f.write(str(self.cds.pulley_location) + "\n")
        f.write(str(self.cds.pp_location) + "\n")
        f.write(str(nContact) + " Contacts\n")
        f.close()
        while cl1_2test > init_cable_length[1] - shorten_dist_max:
            cl_list = self.cds.initial_cable_length.copy()
            cl_list[1] = cl1_2test
            print("generate ws for cable length: ", cl_list)
            cl_return, cable_tension_rounded, contactConfig = self.generate_onews_length(cl_list, ws_file, nContact)
            cl2_2test = cl_return[2]
            while cl2_2test > init_cable_length[2] - shorten_dist_max:
                cl2_2test -= stepSize
                cl_list = self.cds.initial_cable_length.copy()
                cl_list[1] = cl1_2test
                cl_list[2] = cl2_2test
                cl_return, cable_tension_rounded, contactConfig = self.generate_onews_length(cl_list, ws_file, nContact)
                if cl_return[1] < (cl1_2test - 0.001):
                    break
            cl1_2test -= stepSize

    def generate_contact_ws_simple(self, ws_file):
        init_cable_length = self.cds.initial_cable_length.copy()
        shorten_dist_max = 0.06 # maximum shorten distance is 10 cm
        stepSize = 0.003 # every 1mm
        cltest = init_cable_length[1]
        nContact = len(self.cds.contact_vertices)
        # open the file, write cds pulley location, cds pullpos
        f = open(ws_file, "w")
        f.write(str(self.cds.pulley_location) + "\n")
        f.write(str(self.cds.pp_location) + "\n")
        f.write(str(nContact) + " Contacts\n")
        f.close()
        while cltest > init_cable_length[1] - shorten_dist_max:
            cl_list = self.cds.initial_cable_length.copy()
            cltest -= stepSize
            cl_list[1] = cltest
            cl_list[2] = cltest
            print("generate ws for cable length: ", cl_list)
            cl_return, cable_tension_rounded, contactConfig = self.generate_onews_length(cl_list, ws_file, nContact)

            

    def get_contact_config_kineto(self,tree, point_cloud, B_spline_surface, cable_tension_list, initial_vert):
        # calculate the contact configuration
        # return the contact configuration
        H_mat = np.zeros((3*self.cds.nMoving, self.cds.nCable))
        tol = 1e-6
        q = np.zeros((3*self.cds.nMoving, 1))
        for i in range(self.cds.num_vert):
            idx_moving = self.cds.idxAll_2_idxMoving[i]
            if idx_moving != -1:
                q[3*idx_moving:3*idx_moving+3, 0] = initial_vert[i]
        q_last = q.copy()
        cable_tensions = np.array(cable_tension_list).reshape((self.cds.nCable, 1))
        grativity_vec_moving = np.zeros((3*self.cds.nMoving, 1))
        for i in range(self.cds.nMoving):
            idx = self.cds.idxMoving_2_idxAll[i]
            grativity_vec_moving[3*i:3*i+3, 0] = self.cds.gravity_vec[3*idx:3*idx+3, 0]
        f_contact_all = self.get_f_contact(tree, point_cloud, B_spline_surface, self.cds.cur_vertices)
        
        f_contact = np.zeros((3*self.cds.nMoving, 1))
        for i in range(self.cds.nMoving):
            idx = self.cds.idxMoving_2_idxAll[i]
            for j in range(3):
                f_contact[3*i+j, 0] = f_contact_all[3*idx+j, 0]
        max_iter = 100
        for i in range(max_iter):
            q_last = q.copy()
            R_list_1212 = self.cal_rotation_fem(q)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_tilde, K_tilde_vec2add = self.assemble_K_tilde(Ke_list)
            f0_tilde = self.assemble_f0_tilde(Ke0_list)
            H_mat = np.zeros((3*self.cds.nMoving, self.cds.nCable))
            for j in range(self.cds.nCable):
                idx_pp_moving = self.cds.idxAll_2_idxMoving[self.cds.pp_idx[j]]
                unit_vec = self.cds.pulley_location[j,:] - q[3*idx_pp_moving:3*idx_pp_moving+3].reshape((3,))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                H_mat[3*idx_pp_moving:3*idx_pp_moving+3, j] = unit_vec
            f_contact = self.get_f_contact_kineto(tree, point_cloud, B_spline_surface, q)
            vecB = H_mat @ cable_tensions + K_tilde_vec2add + f0_tilde + grativity_vec_moving + f_contact
            # vecB = K_tilde_vec2add + f0_tilde + grativity_vec_moving
            q = np.linalg.solve(K_tilde, vecB)
            diff = np.linalg.norm(q - q_last)
            print("iter: ", i, "diff: ", diff)
            if diff < tol and i > 2:
                break
        
        cur_vert = self.cds.vertices.copy()
        # update current vertices of cds
        for i in range(self.cds.nMoving):
            cur_vert[self.cds.idxMoving_2_idxAll[i],:] = q[3*i:3*i+3].reshape((3,))
        # f_residule = K_tilde @ q - vecB
        return cur_vert, f_contact

    def get_contact_config_dynamic(self, tree, point_cloud, B_spline_surface, cable_tension, starting_vert):
        Q_a = np.zeros((3*self.cds.num_vert, 1))
        for i in range(self.cds.num_vert):
            for j in range(3):
                Q_a[3*i+j, 0] = starting_vert[i][j]
        Q_a_last = np.zeros((3*self.cds.num_vert, 1))
        Q_ad = np.zeros((3*self.cds.num_vert, 1))
        t_a = 0.0
        H_mat = np.zeros((self.cds.nCable, 3*self.cds.num_vert))
        t_total = 10.0
        h = 0.1
        tol = 1e-8
        while t_a < t_total:
            Q_a_last = Q_a.copy()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + h * self.cds.W_mat @ K_mat 
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.cds.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.cds.gravity_vec)
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.cds.num_vert, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            for j in range(self.cds.nCable):
                idx_pp = self.cds.pp_idx[j]
                unit_vec = self.cds.pulley_location[j,:] - Q_free[3*idx_pp:3*idx_pp+3].reshape((3,))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                H_mat[j, 3*idx_pp:3*idx_pp+3] = unit_vec
            f_contact_all = self.get_f_contact_dynamic(tree, point_cloud, B_spline_surface, Q_free)
            dv_cor = A_inv @ self.cds.W_mat @ ((H_mat.T @ cable_tension).reshape((3*self.cds.num_vert, 1)) + f_contact_all)
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            diff = np.linalg.norm(Q_a - Q_a_last)
            print("cur_time: ", t_a, "diff: ", diff)
            if diff < tol:
                break
        vert_contact_dynamic = self.cds.cur_vertices.copy()
        for i in range(self.cds.num_vert):
            vert_contact_dynamic[i] = Q_a[3*i:3*i+3].reshape((3,))
        return vert_contact_dynamic, f_contact_all

    def get_contact_config_length(self, tree, point_cloud, B_spline_surface, cable_length_list, starting_vert):
        xzrange =  (point_cloud[:, 0].min(), point_cloud[:, 0].max(), point_cloud[:, 2].min(), point_cloud[:, 2].max())
        Q_a = np.zeros((3*self.cds.num_vert, 1))
        for i in range(self.cds.num_vert):
            for j in range(3):
                Q_a[3*i+j, 0] = starting_vert[i][j]
        Q_a_last = np.zeros((3*self.cds.num_vert, 1))
        Q_ad = np.zeros((3*self.cds.num_vert, 1))
        t_a = 0.0
        H_free = np.zeros((self.cds.nCable, 3*self.cds.num_vert))
        phi_Qfree = np.zeros((self.cds.nCable, 1))
        t_total = 20.0
        h = 0.1
        tol = 1e-8
        while t_a < t_total:
            Q_a_last = Q_a.copy()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + (h + 1) * self.cds.W_mat @ K_mat 
            A_inv = np.linalg.inv(A_mat)
            f_contact_all = self.get_f_contact_dynamic(tree, point_cloud, B_spline_surface, Q_a, xzrange)
            b_vec = self.cds.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.cds.gravity_vec + f_contact_all - K_mat @ Q_ad)
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.cds.num_vert, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            for i in range(self.cds.nCable):
                idx_pp = self.cds.pp_idx[i]
                unit_vec = (self.cds.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                phi_Qfree[i] = cable_length_list[i] - np.linalg.norm(self.cds.pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
            lcp_Mmat = h * H_free @ self.cds.W_mat @ A_inv @ H_free.T
            # check if lcp_Mmat is PD
            M_is_PD = np.all(np.linalg.eigvals(lcp_Mmat) > 0)
            lcp_q = phi_Qfree.reshape((self.cds.nCable,))
            lcp_sol = qe.optimize.lcp_lemke(lcp_Mmat, lcp_q)
            if not lcp_sol.success:
                print("lcp failed: ")
                break
            cable_tension = lcp_sol.z
            # print("cur time: ", t_a, "cable_tension: ", cable_tension, "lcp_q: ", lcp_q.flatten(), "M_is_PD: ", M_is_PD)
            cable_tension.reshape((self.cds.nCable, 1))
            dv_cor = A_inv @ (self.cds.W_mat @ H_free.T @ cable_tension).reshape((3*self.cds.num_vert, 1))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            if np.linalg.norm(Q_a - Q_a_last) < 1e-3:
                break
        vert_length = self.cds.cur_vertices.copy()
        for i in range(self.cds.num_vert):
            vert_length[i] = Q_a[3*i:3*i+3].reshape((3,))
        return vert_length, cable_tension.flatten(), f_contact_all

    def get_f_contact_kineto(self, tree, point_cloud, B_spline_surface, q):
        f_contact = np.zeros((3*self.cds.nMoving, 1))
        for contact_idx in self.cds.contact_vertices:
            contact_idx_moving = self.cds.idxAll_2_idxMoving[contact_idx]
            if contact_idx_moving == -1:
                continue
            q_query = q[3*contact_idx_moving:3*contact_idx_moving+3].reshape((3,))
            is_inside, distance, unit_direction, closest_point = min_distance_and_direction(tree, point_cloud, B_spline_surface, q_query)
            if is_inside:
                f_contact[3*contact_idx_moving:3*contact_idx_moving+3, 0] = self.stiffness_arm * distance * unit_direction
        return f_contact

    def get_f_contact(self, tree, point_cloud, B_spline_surface, cur_vert):
        f_contact_all = np.zeros((3*self.cds.num_vert, 1))
        for contact_idx in self.cds.contact_vertices:
            q = cur_vert[contact_idx]
            is_inside, distance, unit_direction, closest_point = min_distance_and_direction(tree, point_cloud, B_spline_surface, q)
            if is_inside:
                f_contact_all[3*contact_idx:3*contact_idx+3, 0] = self.stiffness_arm * distance * unit_direction
        return f_contact_all
    
    def get_f_contact_dynamic(self, tree, point_cloud, B_spline_surface, Q_free, xzrange):
        f_contact_all = np.zeros((3*self.cds.num_vert, 1))
        for contact_idx in self.cds.contact_vertices:
            q = Q_free[3*contact_idx:3*contact_idx+3].reshape((3,))
            is_inside, distance, unit_direction, closest_point = min_distance_and_direction_v2(tree, point_cloud, B_spline_surface, q, xzrange)
            if is_inside:
                f_contact_all[3*contact_idx:3*contact_idx+3, 0] = self.stiffness_arm * distance * unit_direction
        return f_contact_all
    
    def get_f_contact_nonlinOpt(self, tree, point_cloud, B_spline_surface, Q_free, xzrange, vertical_moving):
        f_contact_all = np.zeros((3*self.cds.num_vert, 1))
        for contact_idx in self.cds.contact_vertices:
            q = Q_free[3*contact_idx:3*contact_idx+3].reshape((3,))
            q[1] = q[1] - vertical_moving
            is_inside, distance, unit_direction, closest_point = min_distance_and_direction_v2(tree, point_cloud, B_spline_surface, q, xzrange)
            if is_inside:
                f_contact_all[3*contact_idx:3*contact_idx+3, 0] = self.stiffness_arm * distance * unit_direction
        return f_contact_all

    def read_ws_file(self, ws_file):
        ws_dic = {"cable_length": [], "cable_tension": [], "contact_config": []}
        # the first 2XnCable liness are info of pulley location and pp_location, skip them
        with open(ws_file, "r") as f:
            for i in range(2 * self.cds.nCable):
                f.readline()
            nContact = int(f.readline().split()[0])
            while True:
                cl_line = f.readline()
                if not cl_line:
                    break
                ws_dic["cable_length"].append(eval(cl_line))
                ct_line = f.readline()
                ws_dic["cable_tension"].append(eval(ct_line))
                # the next nContact lines are contact config, read into an np array
                contact_config = np.zeros((nContact, 3))
                for i in range(nContact):
                    # each line is a contact config in format [x y z]
                    pos_line = f.readline()
                    clean_posline = pos_line.replace('[', '').replace(']', '')
                    contact_config[i] = [float(x) for x in clean_posline.split()]
                ws_dic["contact_config"].append(contact_config)
        return ws_dic

    def find_best_match_ws(self, ws_dic, B_spline_surface, xzranges):
        min_x, max_x, min_z, max_z = xzranges
        n_ee = len(ws_dic["cable_length"])
        nContact = ws_dic['contact_config'][0].shape[0]
        vertical_moving_list = []
        diff_list_norm_final = []
        for i in range(n_ee):
            contact_pts_original = ws_dic["contact_config"][i].copy()
            contact_pts = ws_dic["contact_config"][i].copy()
            diff_list = [0.0 for _ in range(nContact)]
            for _ in range(10): # iterate 10 times
                for j in range(nContact):
                    x, y, z = contact_pts[j]
                    u_this = (x-min_x)/(max_x-min_x)
                    v_this = (z-min_z)/(max_z-min_z)
                    Y_fitted = interpolate.bisplev(u_this, v_this, B_spline_surface)
                    diff_list[j] = Y_fitted - y
                mean_diff = np.mean(diff_list)
                contact_pts[:, 1] = contact_pts[:, 1] + mean_diff
            vertical_moving_list.append(np.mean(contact_pts_original[:,1]) - np.mean(contact_pts[:,1]))
            final_diff = np.linalg.norm(diff_list)
            diff_list_norm_final.append(final_diff)
        # find the index with smallest diff
        best_match_idx = np.argmin(diff_list_norm_final)
        best_cl = ws_dic["cable_length"][best_match_idx]
        best_ct = ws_dic["cable_tension"][best_match_idx]
        best_contact_config = ws_dic["contact_config"][best_match_idx]
        vertical_moving = vertical_moving_list[best_match_idx]
        return best_cl, best_ct, best_contact_config, vertical_moving

    def find_final_contact_cmd(self, best_cl, vertical_moving, pts_augmented, tree, B_spline_surface, xzrange):
        final_contact_cmd = self.cds.initial_cable_length.copy()
        vert_length_noContact, cable_tension_noContact, R_list = self.fk_length_allCableTensioned(best_cl, [np.eye(3) for _ in range(cds.num_tet + cds.nCable)])
        tar_distance = 0.0 # 1mm inside the surface
        coeff_nonacontact = 10
        coeff_contact = 0.5
        def contact_rating(x): # x is an optimization variable, x[0:nCable] is cable length, x[nCable] is vertical moving
            cl = x[0: self.cds.nCable]
            v_m = x[self.cds.nCable]
            pts_augmented_copy = pts_augmented.copy()
            pts_augmented_copy[:, 1] = pts_augmented_copy[:, 1] + v_m
            Q_a = np.zeros((3*self.cds.num_vert, 1))
            for i in range(self.cds.num_vert):
                for j in range(3):
                    Q_a[3*i+j, 0] = vert_length_noContact[i][j]
            Q_a_last = np.zeros((3*self.cds.num_vert, 1))
            Q_ad = np.zeros((3*self.cds.num_vert, 1))
            t_a = 0.0
            H_free = np.zeros((self.cds.nCable, 3*self.cds.num_vert))
            phi_Qfree = np.zeros((self.cds.nCable, 1))
            t_total = 20.0
            h = 0.1
            while t_a < t_total:
                Q_a_last = Q_a.copy()
                R_list_1212 = self.cal_rotation_fkLength(Q_a)
                Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
                K_mat = self.assemble_K(Ke_list)
                f0 = self.assemble_f0(Ke0_list)
                A_mat = (1.0/h)*np.eye(3*self.cds.num_vert) + (h + 1) * self.cds.W_mat @ K_mat 
                A_inv = np.linalg.inv(A_mat)
                f_contact_all = self.get_f_contact_nonlinOpt(tree, pts_augmented, B_spline_surface, Q_a, xzrange, v_m)
                b_vec = self.cds.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.cds.gravity_vec + f_contact_all - K_mat @ Q_ad)
                dv_free = A_inv @ b_vec
                dv_free.reshape((3*self.cds.num_vert, 1))
                Q_free = Q_a + h * Q_ad + h * dv_free
                for i in range(self.cds.nCable):
                    idx_pp = self.cds.pp_idx[i]
                    unit_vec = (self.cds.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                    unit_vec = unit_vec / np.linalg.norm(unit_vec)
                    # print("unit_vec: ", unit_vec)
                    phi_Qfree[i] = cl[i] - np.linalg.norm(self.cds.pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                    H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
                lcp_Mmat = h * H_free @ self.cds.W_mat @ A_inv @ H_free.T
                # check if lcp_Mmat is PD
                M_is_PD = np.all(np.linalg.eigvals(lcp_Mmat) > 0)
                lcp_q = phi_Qfree.reshape((self.cds.nCable,))
                lcp_sol = qe.optimize.lcp_lemke(lcp_Mmat, lcp_q)
                if not lcp_sol.success:
                    print("lcp failed: ")
                    break
                cable_tension = lcp_sol.z
                cable_tension.reshape((self.cds.nCable, 1))
                dv_cor = A_inv @ (self.cds.W_mat @ H_free.T @ cable_tension).reshape((3*self.cds.num_vert, 1))
                dv = dv_free + dv_cor
                Q_ad = Q_ad + dv
                Q_a = Q_a + h * Q_ad
                t_a += h
                if np.linalg.norm(Q_a - Q_a_last) < 1e-3:
                    break
            final_rating = 0.0
            Qa_test = Q_a.copy()
            for contact_idx in self.cds.contact_vertices:
                q = Qa_test[3*contact_idx:3*contact_idx+3].reshape((3,))
                q[1] = q[1] - v_m
                is_inside, distance, unit_direction, closest_point = min_distance_and_direction_v2(tree, pts_augmented, B_spline_surface, q, xzrange)
                if is_inside:
                    final_rating += coeff_contact * distance**2
                else:
                    final_rating += coeff_nonacontact * distance**2
            print("run for x: ", x, "final_rating: ", final_rating)
            return final_rating

        x0 = best_cl + [vertical_moving]
        res = minimize(contact_rating, x0, method='SLSQP', tol = 1e-6, options={'eps': 1e-3, 'maxiter': 1000})
        final_contact_cmd = res.x
        return final_contact_cmd

    def find_best_match_ws_encureContact(self, ws_dic, B_spline_surface, xzranges):
        min_x, max_x, min_z, max_z = xzranges
        num_ws = len(ws_dic["cable_length"])
        nContact = ws_dic['contact_config'][0].shape[0]
        vertical_moving_list = []
        diff_list_norm_final = []
        for i in range(num_ws):
            contact_pts_original = ws_dic["contact_config"][i].copy()
            contact_pts = ws_dic["contact_config"][i].copy()
            diff_list = [0.0 for _ in range(nContact)]
            for j in range(nContact):
                x, y, z = contact_pts[j]
                u_this = (x-min_x)/(max_x-min_x)
                v_this = (z-min_z)/(max_z-min_z)
                Y_fitted = interpolate.bisplev(u_this, v_this, B_spline_surface)
                diff_list[j] = Y_fitted - y
            max_diff = np.max(diff_list)
            contact_pts[:, 1] = contact_pts[:, 1] + max_diff
            vertical_moving_list.append(np.mean(contact_pts_original[:,1]) - np.mean(contact_pts[:,1]))
            for j in range(nContact):
                x, y, z = contact_pts[j]
                u_this = (x-min_x)/(max_x-min_x)
                v_this = (z-min_z)/(max_z-min_z)
                Y_fitted = interpolate.bisplev(u_this, v_this, B_spline_surface)
                diff_list[j] = Y_fitted - y
            final_diff = np.linalg.norm(diff_list)
            diff_list_norm_final.append(final_diff)
        # find the index with smallest diff
        best_match_idx = np.argmin(diff_list_norm_final)
        best_cl = ws_dic["cable_length"][best_match_idx]
        best_ct = ws_dic["cable_tension"][best_match_idx]
        best_contact_config = ws_dic["contact_config"][best_match_idx]
        vertical_moving = vertical_moving_list[best_match_idx]
        return best_cl, best_ct, best_contact_config, vertical_moving
    
    # code for checking intersection between cable and tetrahedron
    def check_intersection(self, pp_loc, check_mesh):
        # Möller-Trumbore intersection algorithm.
        # pp_loc is the pull points location, check mesh is np array of vertices of the mesh
        is_intersect = [True for _ in range(self.cds.nCable)]
        tol = 1e-6
        ray_list = np.zeros((self.cds.nCable, 3))
        intesect_array = np.zeros((self.cds.nCable, 3))
        length_list = [0.0 for _ in range(self.cds.nCable)]
        for i in range(self.cds.nCable):
            length_list[i] = np.linalg.norm(self.cds.pulley_location[i] - pp_loc[i])
        for i in range(self.cds.nCable):
            ray_list[i] = (self.cds.pulley_location[i] - pp_loc[i])/length_list[i]

        for j in range(self.cds.nCable):
            ray_vec = ray_list[j,:]
            for i in range(self.cds.triangle_list.shape[0]):
                is_intersect[j] = True
                v0 = check_mesh[self.cds.triangle_list[i, 0],:]
                v1 = check_mesh[self.cds.triangle_list[i, 1],:]
                v2 = check_mesh[self.cds.triangle_list[i, 2],:]
                # Calculate the normal vector of the triangle
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                p_vec = np.cross(ray_vec, edge2)
                det = np.dot(edge1, p_vec)
                if abs(det) < tol:
                    is_intersect[j] = False
                inv_det = 1.0 / det
                t_vec = pp_loc[j] - v0
                u = inv_det * np.dot(t_vec, p_vec)
                if u < 0.0 or u > 1.0:
                    is_intersect[j] = False
                q_vec = np.cross(t_vec, edge1)
                v = inv_det * np.dot(ray_vec, q_vec)
                if v < 0.0 or u + v > 1.0:
                    is_intersect[j] = False
                t = inv_det * np.dot(edge2, q_vec)
                if t < tol or t > length_list[j]+tol:
                    is_intersect[j] = False
                # put the intersection point into the array
                if is_intersect[j]:
                    intesect_array[j,:] = pp_loc[j] + t * ray_vec
                    break
            print("cable: ", j, "is_intersect: ", is_intersect[j], "intersect_point: ", intesect_array[j,:])
        return is_intersect, intesect_array        
        

if __name__ == "__main__":
    pp_location = np.array([[0.05352376, -0.0127263,   0. ], [0.07471047, 0.02281154, 0.0073504], [0.07471047, 0.02281154, 0.0576496], [0.05352376, -0.0127263 ,  0.065 ]])
    pulley_location = np.array([[0.07, -0.10, 0],[0.13, -0.10, 0],[0.13, -0.10, 0.07], [0.07, -0.10, 0.07]])
    cds = cableDrivenSurface("models/myPalm/palm_v4/palm_v4.tet", pp_location, pulley_location)
    deformer = cdsDeformer(cds)
    ws_dic = deformer.read_ws_file("models/myPalm/palm_v4/ws.txt")
    pp_location = np.array([[57.61, 8.96, 4.5], [65.12, 46.49, 8.00], [65.12, 46.49, 57], [57.61, 8.96, 60.5]])*1e-3
    pulley_location = np.array([[50, -122.32, 2.5], [100, -122.32, 2.5],[100, -122.32, 62.5], [50, -122.32, 62.5]])
    # pulley_location[:, 0] += 10
    pulley_location *= 1e-3
    cds = cableDrivenSurface("models/myPalm/palm_size3/palm_size3.tet", pp_location, pulley_location)
    deformer = cdsDeformer(cds)
    pp_loc = np.zeros((cds.nCable, 3))
    for i in range(cds.nCable):
        pp_loc[i,:] = cds.cur_vertices[cds.pp_idx[i],:]
    print("start checking intersection")
    start_t = time.time()
    is_intersect, intersect_array = deformer.check_intersection(pp_loc, cds.vertices)
    end_t = time.time()
    print("inter_check_time: ", end_t - start_t)
    print("intersect_array: ", intersect_array)
    print("is_intersect: ", is_intersect)
    # deformer.generate_contact_ws("models/myPalm/palm_v4/ws.txt")
    pt_filename = "./touch_region1.ply"
    # read the point cloud into an array
    points = np.loadtxt(pt_filename)*1e-3
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    zmin = np.min(points[:, 2])
    zmax = np.max(points[:, 2])
    points[:, 0] = points[:, 0] - xmin+0.02
    points[:, 1] = points[:, 1] - ymin
    points[:, 2] = points[:, 2] - zmin
    tree = KDTree(points)
    # Fit B-spline surface
    B_spline_surface = fit_bspline_surface(points)
    # pts_augmented = augument_point_cloud(points, B_spline_surface)
    # tree_augmented = KDTree(pts_augmented)
    xzrange = (points[:, 0].min(), points[:, 0].max(), points[:, 2].min(), points[:, 2].max())
    best_cl, best_ct, best_contact_config, vertical_moving = deformer.find_best_match_ws_encureContact(ws_dic, B_spline_surface, xzrange)
    best_cl, best_ct, best_contact_config, vertical_moving = deformer.find_best_match_ws(ws_dic, B_spline_surface, xzrange)
    print("vertical_moving: ", vertical_moving)
    best_points = points.copy()
    best_points[:,1] = points[:,1] + vertical_moving
    best_B_spline_surface = fit_bspline_surface(best_points)
    best_pts_augmented = augument_point_cloud(best_points, B_spline_surface)
    best_tree_augmented = KDTree(best_pts_augmented)
    icl = cds.initial_cable_length
   
    vert_length_noContact, cable_tension_noContact, R_list = deformer.fk_length_startMesh(best_cl, cds.vertices)
    cds.cur_vertices = vert_length_noContact
    # vert_length, cable_tension, f_contact_all = deformer.get_contact_config_length(best_tree_augmented, best_pts_augmented, best_B_spline_surface, cable_command, vert_length_noContact)
    # cds.cur_vertices = vert_length
    # print norm of f_contact_all if it is not 0
    # for i in range(cds.num_vert):
    #     if np.linalg.norm(f_contact_all[3*i:3*i+3]) > 1e-6:
    #         print("f_contact_all[", i, "]: ", np.linalg.norm(f_contact_all[3*i:3*i+3]))
    # final_cable_length = cds.initial_cable_length.copy()
    # for i in range(cds.nCable):
    #     idx_pp = cds.pp_idx[i]
    #     final_cable_length[i] = np.linalg.norm(vert_length_noContact[idx_pp] - cds.pulley_location[i])
    # print("final_cable_length: ", final_cable_length)
    # print("cable length diff: ", [icl- fcl for icl, fcl in zip(icl, final_cable_length)])

    cds.draw_cur_mesh_touchRegion_noContactConfig(best_pts_augmented, best_B_spline_surface, vert_length_noContact)
    # cds.draw_cur_mesh_and_touchRegion(best_pts_augmented, best_B_spline_surface, f_contact_all)
    # cds.draw_cur_mesh_w_initial_mesh() 
    # cds.draw_mesh_w_intersection(intersect_array, cds.vertices)
    
