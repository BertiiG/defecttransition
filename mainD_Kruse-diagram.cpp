//
// Created by Abhinav Singh on 15.03.20.
//
//#define SE_CLASS1
#include "config.h"
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_VECTOR_SIZE 50
#include <iostream>
#include "DCPSE/DCPSE_op/DCPSE_op.hpp"
#include "DCPSE/DCPSE_op/DCPSE_Solver.hpp"
#include "Operators/Vector/vector_dist_operators.hpp"
#include "Vector/vector_dist_subset.hpp"
#include "DCPSE/DCPSE_op/EqnsStruct.hpp"
#include "OdeIntegrators/OdeIntegrators.hpp"

constexpr int x = 0;
constexpr int y = 1;
constexpr int POLARIZATION= 0,VELOCITY = 1, VORTICITY = 2, NORMAL = 3,PRESSURE = 4, STRAIN_RATE = 5, STRESS = 6, MOLFIELD = 7, DPOL = 8, DV = 9, VRHS = 10, F1 = 11, F2 = 12, F3 = 13, F4 = 14, F5 = 15, F6 = 16, V_T = 17, DIV = 18, DELMU = 19, HPB = 20, FE = 21, R = 22, PID = 23, POLD = 24;


double eta = 10.0;
double nu = 3.0;
double gama = 10.0;
double zeta = -1.0;
double lambda = 1.0;
double Ks = 1.0;
double chi2 = -5.;
double chi4 = -chi2/2.;
double k = .0;
double zetadelmu;
double dkK;
double delmu;
double Kb;
double sum2;
double sum3;
double anglesum;
double anglesum_old;

int wr_f;
int wr_at;
double V_err_eps;
timer gt;
timer tt2;

int flag = 1;
double steady_tol=1e-9;

void *vectorGlobal=nullptr,*vectorGlobal_bulk=nullptr,*vectorGlobal_boundary=nullptr;
const openfpm::vector<std::string>
PropNAMES={"00-Polarization","01-Velocity","02-Vorticity","03-Normal","04-Pressure","05-StrainRate","06-Stress","07-MolecularField","08-DPOL","09-DV","10-VRHS","11-f1","12-f2","13-f3","14-f4","15-f5","16-f6","17-V_T","18-DIV","19-DELMU","20-HPB","21-FrankEnergy","22-R","23-particleID","24-P_old"};
typedef aggregate<VectorS<2, double>,VectorS<2, double>,double[2][2],VectorS<2, double>,double,double[2][2],double[2][2],VectorS<2, double>,VectorS<2, double>,VectorS<2, double>,VectorS<2, double>,double,double,double,double,double,double,VectorS<2, double>,double,double,double,double,double,int,VectorS<2, double>> Activegels;
typedef vector_dist_ws<2, double,Activegels> vector_type;
typedef vector_dist_subset<2, double, Activegels> vector_type2;

openfpm::vector<aggregate<vect_dist_key_dx[2]>> CorrVec;        // vector to store the Ids for the Neumann BC, 0: boundary 1: bulk

//Functor to Compute RHS of the time derivative of the polarity
template<typename DX,typename DY,typename DXX,typename DXY,typename DYY>
struct PolarEv
{
    DX &Dx;
    DY &Dy;
    DXX &Dxx;
    DXY &Dxy;
    DYY &Dyy;
    //Constructor
    PolarEv(DX &Dx,DY &Dy,DXX &Dxx,DXY &Dxy,DYY &Dyy):Dx(Dx),Dy(Dy),Dxx(Dxx),Dxy(Dxy),Dyy(Dyy)
    {}

    void operator()( const state_type_2d_ofp &X , state_type_2d_ofp &dxdt , const double t ) const
    {

        vector_type &Particles= *(vector_type *) vectorGlobal;
        vector_type2 &Particles_bulk= *(vector_type2 *) vectorGlobal_bulk;
        vector_type2 &Particles_boundary= *(vector_type2 *) vectorGlobal_boundary;


        auto & v_cl=create_vcluster();
        auto & bulk = Particles_bulk.getIds();
        auto & boundary = Particles_boundary.getIds();


        auto Dyx=Dxy;
        auto Pol=getV<POLARIZATION>(Particles);
        auto Pol_bulk=getV<POLARIZATION>(Particles_bulk);
        auto V = getV<VELOCITY>(Particles);
        auto Vbdry = getV<VELOCITY>(Particles_boundary);
        auto h = getV<MOLFIELD>(Particles);
        auto u = getV<STRAIN_RATE>(Particles);
        auto dPol = getV<DPOL>(Particles);
        auto W = getV<VORTICITY>(Particles);
        //auto delmu = getV<DELMU>(Particles);
        //auto H_p_b = getV<HPB>(Particles);
        auto r = getV<R>(Particles);
        auto dPol_bulk = getV<DPOL>(Particles_bulk);

        auto sigma = getV<STRESS>(Particles);
        // auto asigma = getV<ASTRESS>(Particles);
        auto FranckEnergyDensity = getV<FE>(Particles);
        auto f1 = getV<F1>(Particles);
        auto f2 = getV<F2>(Particles);
        auto f3 = getV<F3>(Particles);
        auto f4 = getV<F4>(Particles);
        auto f5 = getV<F5>(Particles);
        // auto f6 = getV<F6>(Particles); //use for old FranckEnergyDensity
        auto dV = getV<DV>(Particles);
        //auto g = getV<NORMAL>(Particles);
        auto P = getV<PRESSURE>(Particles);
        auto P_bulk = getV<PRESSURE>(Particles_bulk);
        auto RHS = getV<VRHS>(Particles);
        auto RHS_bulk = getV<VRHS>(Particles_bulk);
        auto div = getV<DIV>(Particles);
        auto V_t = getV<V_T>(Particles);

        Pol[x]=X.data.get<0>();
        Pol[y]=X.data.get<1>();
        // impose Neumann BC for polarisation given the boundary and bulk pairs
        for (int j = 0; j < CorrVec.size(); j++) {
                auto p_out = CorrVec.get<0>(j)[0];
                auto p = CorrVec.get<0>(j)[1];
                Particles.getProp<0>(p_out)=Particles.getProp<0>(p);
                Particles.getProp<F3>(p_out)=Particles.getProp<F3>(p);
            }
        //Particles.ghost_get<0>(SKIP_LABELLING);

        Particles.ghost_get<POLARIZATION, F3>(SKIP_LABELLING);

        eq_id x_comp, y_comp;
        x_comp.setId(0);
        y_comp.setId(1);

        int n = 0,nmax = 300,errctr = 0, Vreset = 0;
        double V_err = 1, V_err_old,sum, sum1;
        timer tt;
        tt.start();
        petsc_solver<double> solverPetsc;
        solverPetsc.setSolver(KSPGMRES);
        //solverPetsc.setPreconditioner(PCLU);
        // calculate erickson stress
        sigma[x][x] = - Ks * Dx(Pol[x])*Dx(Pol[x]) - Ks * Dy(Pol[x])*Dx(Pol[y])
                      + Kb * Dy(Pol[x])*Dx(Pol[y]) - Kb * Dx(Pol[y])*Dx(Pol[y])
                      - k * Dx(Pol[x])
                      ;
        sigma[x][y] = - Ks * Dx(Pol[y])*Dx(Pol[x]) - Ks * Dy(Pol[y])*Dx(Pol[y])
                      + Kb * Dx(Pol[y])*Dx(Pol[x]) - Kb * Dy(Pol[x])*Dx(Pol[x])
                      - k * Dx(Pol[y])
                      ;
        sigma[y][x] = - Ks * Dx(Pol[x])*Dy(Pol[x]) - Ks * Dy(Pol[x])*Dy(Pol[y])
                      + Kb * Dy(Pol[x])*Dy(Pol[y]) - Kb * Dx(Pol[y])*Dy(Pol[y])
                      - k  * Dy(Pol[x])
                      ;
        sigma[y][y] = - Ks * Dx(Pol[y])*Dy(Pol[x]) - Ks * Dy(Pol[y])*Dy(Pol[y])
                      + Kb * Dx(Pol[y])*Dy(Pol[x]) - Kb * Dy(Pol[x])*Dy(Pol[x])
                      - k * Dy(Pol[y])
                      ;
        Particles.ghost_get<STRESS>(SKIP_LABELLING);

        Particles.deleteGhost();
        Particles.write_frame("PolarTest",1);
        Particles.ghost_get<POLARIZATION>();


        // calulate FranckEnergyDensity
        FranckEnergyDensity = (Ks/2.0) * ((Dx(Pol[x]) * Dx(Pol[x])) +
                              (Dy(Pol[y]) * Dy(Pol[y])) + 2. * Dx(Pol[y]) * Dy(Pol[x])) +
                              (Kb/2.0) * ((Dx(Pol[y]))*(Dx(Pol[y])) - 2. * (Dx(Pol[y]))*(Dy(Pol[x])) + (Dy(Pol[x]))*(Dy(Pol[x])) ) +
                              chi2/2. * (Pol[x] * Pol[x] + Pol[y] * Pol[y]) +
                              chi4/2. * (Pol[x] * Pol[x] * Pol[x] * Pol[x] + Pol[y] * Pol[y] * Pol[y] * Pol[y] + 2 * Pol[x] * Pol[x] * Pol[y] * Pol[y])
                              + k * (Dx(Pol[x]) + Dy(Pol[y]))
                              ;
        Particles.ghost_get<FE>(SKIP_LABELLING);



        f1 = Kb * Dyy(Pol[x]) - Kb * Dyx(Pol[y]) +
             Ks * Dxx(Pol[x]) + Ks * Dyx(Pol[y]) -
             (chi2 * Pol[x] + 2. * chi4 * Pol[x] * Pol[x] * Pol[x] + 2. * chi4 * Pol[y] * Pol[y] * Pol[x]);
        f2 = Kb * Dxx(Pol[y]) - Kb * Dxy(Pol[x]) +
             Ks * Dyy(Pol[y]) + Ks * Dxy(Pol[x]) -
             (chi2 * Pol[y] + 2. * chi4 * Pol[y] * Pol[y] * Pol[y] + 2. * chi4 * Pol[x] * Pol[x] * Pol[y]);

        Particles.ghost_get<F1, F2>(SKIP_LABELLING);
        texp_v<double> Dxf1 = Dx(f1),Dxf2 = Dx(f2), Dyf1 = Dy(f1), Dyf2 = Dy(f2);

        //H_perpendicular
        f4 = f2 * Pol[x] - f1 * Pol[y];
        //H_parallel
        f5 = f1 * Pol[x] + f2 * Pol[y];
        Particles.ghost_get<F4, F5>(SKIP_LABELLING);

        // calulate RHS of Stokes equ (without pressure (because pressure correction will be done later)
        dV[x] = - Dx(sigma[x][x]) - Dy(sigma[x][y]) //erickson stress
                - .5 * (f2 * Dy(Pol[x]) + Pol[x] * Dyf2
                        - f1 * Dy(Pol[y]) - Pol[y] * Dyf1)
                + zeta * delmu * ( Pol[x] * Dx(Pol[x]) + Pol[y] * Dy(Pol[x]) + Pol[x] * Dy(Pol[y]))
                - nu/2. * (f1 * Dx(Pol[x]) + Pol[x] * Dxf1
                          + f2 * Dy(Pol[x]) + Pol[x] * Dyf2
                          + f1 * Dy(Pol[y]) + Pol[y] * Dyf1);

        dV[y] = - Dy(sigma[y][y]) - Dx(sigma[y][x]) //erickson STRESS
                - .5 * (f1 * Dx(Pol[y]) + Pol[y] * Dxf1
                        - f2 * Dx(Pol[x]) - Pol[x] * Dxf2)
                + zeta * delmu * ( Pol[y] * Dy(Pol[y]) + Pol[y] * Dx(Pol[x]) + Pol[x] * Dx(Pol[y]))
                - nu/2. * (f2 * Dy(Pol[y]) + Pol[y] * Dyf2
                          + f1 * Dx(Pol[y]) + Pol[y] * Dxf1
                          + f2 * Dx(Pol[x]) + Pol[x] * Dxf2);


        //calculate LHS
        auto Stokes1 = eta * Dxx(V[x]) + eta * Dyy(V[x]);
        auto Stokes2 = eta * Dxx(V[y]) + eta * Dyy(V[y]);

        tt.stop();
        if (v_cl.rank() == 0) {
        std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
        }
        tt.start();
        V_err = 1;
        n = 0;
        errctr = 0;
        if (Vreset == 1) {
            P = 0;
            Vreset = 0;
        }
        P=0;

        // integrate velocity
        Particles.ghost_get<PRESSURE>(SKIP_LABELLING);
        RHS_bulk[x] = dV[x];
        RHS_bulk[y] = dV[y];
        Particles.ghost_get<VRHS>(SKIP_LABELLING);

        // prepare solver
        DCPSE_scheme<equations2d2, vector_type> Solver(Particles);//,options_solver::LAGRANGE_MULTIPLIER
        //impose: impose an operator on a particular particle region to produce th system Ax=b
        Solver.impose(Stokes1, bulk, RHS[0], x_comp);
        Solver.impose(Stokes2, bulk, RHS[1], y_comp);
        Solver.impose(V[x], boundary, 0, x_comp);
        Solver.impose(V[y], boundary, 0, y_comp);
        //std::cout << "P = " << Particles.getProp<PRESSURE>(1) << '\n';
        Solver.solve_with_solver(solverPetsc, V[x], V[y]);
        Particles.ghost_get<VELOCITY>(SKIP_LABELLING);
        div = -(Dx(V[x]) + Dy(V[y]));
        //std::cout << "div =" << Particles.getProp<DIV>(1) << '\n';
        //std::cout << "V = " << Particles.getProp<VELOCITY>(1)[0] << '\n';
        P_bulk = P + 0.00001*div;

        // approximate velocity
        while (V_err >= V_err_eps && n <= nmax) {
            Vbdry=0;
            Particles.ghost_get<PRESSURE>(SKIP_LABELLING);
            //pressure correction
            RHS_bulk[x] = dV[x] + Dx(P);
            RHS_bulk[y] = dV[y] + Dy(P);
            Particles.ghost_get<VRHS>(SKIP_LABELLING);
            Solver.reset_b();
            Solver.impose_b(bulk, RHS[0], x_comp); //update b Seite von Gleichung
            Solver.impose_b(bulk, RHS[1], y_comp);

            Solver.impose_b(boundary, 0, x_comp);
            Solver.impose_b(boundary, 0, y_comp);
            Solver.solve_with_solver_ig(solverPetsc, V[x], V[y]);

            Particles.ghost_get<VELOCITY>(SKIP_LABELLING);
            div = -(Dx(V[x]) + Dy(V[y]));
            P_bulk = P + 0.00001*div;
            // calculate error
            sum = 0;
            sum1 = 0;
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                sum += (Particles.getProp<V_T>(p)[0] - Particles.getProp<VELOCITY>(p)[0]) *
                       (Particles.getProp<V_T>(p)[0] - Particles.getProp<VELOCITY>(p)[0]) +
                       (Particles.getProp<V_T>(p)[1] - Particles.getProp<VELOCITY>(p)[1]) *
                       (Particles.getProp<V_T>(p)[1] - Particles.getProp<VELOCITY>(p)[1]);
                sum1 += Particles.getProp<VELOCITY>(p)[0] * Particles.getProp<VELOCITY>(p)[0] +
                        Particles.getProp<VELOCITY>(p)[1] * Particles.getProp<VELOCITY>(p)[1];
            }
            V_t = V;
            // std::cout << "V_t = " << Particles.getProp<V_T>(1)[0] << '\n';
            // std::cout << "V_t - V = " << Particles.getProp<V_T>(1)[0] - Particles.getProp<VELOCITY>(1)[0] << '\n';
            v_cl.sum(sum);
            v_cl.sum(sum1);
            v_cl.execute();
            sum = sqrt(sum);
            sum1 = sqrt(sum1);
            V_err_old = V_err;
            //if (sum1==0){sum1 = 0;}else{sum1=sum1;}
            // std::cout << "sum = "<< sum << '\n';
            // std::cout << "sum1 = "<< sum1 << '\n';
            V_err = sum / sum1;
            //if (sum1 !=0) {V_err = sum / sum1;} else {V_err = 0;}
            if (V_err > V_err_old || abs(V_err_old - V_err) < 1e-8) {
                errctr++;
            } else {
                errctr = 0;
            }
            if (n > 3) {
                if (errctr > 3) {
                    std::cout << "CONVERGENCE LOOP BROKEN DUE TO INCREASE/VERY SLOW DECREASE IN DIVERGENCE" << std::endl;
                    Vreset = 1;
                    break;
                } else {
                    Vreset = 0;
                }
            }
            n++;
        }
        tt.stop();
        if (v_cl.rank() == 0) {
            std::cout << "Rel l2 cgs err in V = " << V_err << " and took " << tt.getwct() << " seconds with " << n
                      << " iterations."
                      << std::endl;
        }

        Particles.ghost_get<VELOCITY>(SKIP_LABELLING);
        //calculate strain rate
        u[x][x] = Dx(V[x]);
        u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
        u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
        u[y][y] = Dy(V[y]);

        // calculate vorticity
        W[x][x] = 0;
        W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
        W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
        W[y][y] = 0;

        //H_p_b = Pol[x] * Pol[x] + Pol[y] * Pol[y];
        auto it=Particles.getDomainIterator();

        dPol[x] = f1/gama
                  + W[x][y] * Pol[y]
                   - nu * u[x][x] * Pol[x] - nu * u[x][y] * Pol[y]
                   - nu * u[x][x] * Pol[x] - nu * u[y][y] * Pol[x]
                   - (V[x] * Dx(Pol[x]) + V[y] * Dy(Pol[x]))
                   + lambda * delmu * Pol[x]
                   ;

        dPol[y] = f2/gama
                  + W[y][x] * Pol[x]
                   - nu * u[y][x] * Pol[x] - nu * u[y][y] * Pol[y]
                   - nu * u[x][x] * Pol[y] - nu * u[y][y] * Pol[y]
                   - (V[x] * Dx(Pol[y]) + V[y] * Dy(Pol[y]))
                   + lambda * delmu * Pol[y]
                   ;

        dxdt.data.get<0>()=dPol[x];
        dxdt.data.get<1>()=dPol[y];
    }
};

// Functor to calculate velocity and move particles with explicit euler
template<typename DX,typename DY,typename DXX,typename DXY,typename DYY>
struct CalcVelocity
{

    DX &Dx;
    DY &Dy;
    DXX &Dxx;
    DXY &Dxy;
    DYY &Dyy;

    double t_old;
    int ctr;

    //Constructor
    CalcVelocity(DX &Dx,DY &Dy,DXX &Dxx,DXY &Dxy,DYY &Dyy):Dx(Dx),Dy(Dy),Dxx(Dxx),Dxy(Dxy),Dyy(Dyy)
    {
        t_old = 0.0;
        ctr = 0;
    }

    void operator() (state_type_2d_ofp &state, double t)
    {
        vector_type &Particles= *(vector_type *) vectorGlobal;
        vector_type2 &Particles_bulk= *(vector_type2 *) vectorGlobal_bulk;
        vector_type2 &Particles_boundary= *(vector_type2 *) vectorGlobal_boundary;

        auto & v_cl=create_vcluster();
        auto & bulk = Particles_bulk.getIds();
        auto & boundary = Particles_boundary.getIds();


        auto Pol=getV<POLARIZATION>(Particles);
        auto Pol_bulk=getV<POLARIZATION>(Particles_bulk);
        auto dPol = getV<DPOL>(Particles);
        auto FranckEnergyDensity = getV<FE>(Particles);
        auto Pol_old = getV<POLD>(Particles);
        auto f5 = getV<F5>(Particles);
        auto f6 = getV<F6>(Particles);

        gt.stop();
        if (t != 0) {
            Pol[x]=state.data.get<0>();
            Pol[y]=state.data.get<1>();
            for (int j = 0; j < CorrVec.size(); j++) {
                auto p_out = CorrVec.get<0>(j)[0];
                auto p = CorrVec.get<0>(j)[1];
                //std::cout<<p<<":"<<CorrMap[p].getKey()<<" at: "<<v_cl.rank()<<std::endl;
                Particles.getProp<0>(p_out)=Particles.getProp<0>(p);
            }
            Particles.ghost_get<POLARIZATION>(SKIP_LABELLING);
            //auto & Bulk = Particles_bulk.getIds();
            Point<2,double> Xpn,X2pn,xpd1={5.,5.};
            for (int s=0; s<bulk.size(); s++){
              auto p = bulk.get<0>(s);
              Xpn={Particles.getPos(p)[0],Particles.getPos(p)[1]};
              double dist=Xpn.distance(xpd1);
              dist = (dist == 0) ? 1 : dist;
              double p1 = Particles.getProp<POLARIZATION>(s)[x];
              double p2 = Particles.getProp<POLARIZATION>(s)[y];
              //compute normal
              double x1 = (5. - Xpn[0])/dist;
              double x2 = (5. - Xpn[1])/dist;
              double scalar = x1 * p1 + x2 * p2;
              double norm = sqrt(x1*x1 + x2*x2) * sqrt(p1*p1 + p2*p2);
              Particles.getProp<F3>(s) = acos(scalar/norm);
            }
            // for (int l=0; l<boundary.size(); l++){
            //   auto q = boundary.get<0>(t);
            //   X2pn={Particles.getPos(q)[0],Particles.getPos(q)[1]};
            //   double dist=X2pn.distance(xpd1);
            //   dist = (dist == 0) ? 1 : dist;
            //   double q1 = Particles.getProp<POLARIZATION>(l)[x];
            //   double q2 = Particles.getProp<POLARIZATION>(l)[y];
            //   //compute normal
            //   double x1 = (6. - Xpn[0])/dist;
            //   double x2 = (6. - Xpn[1])/dist;
            //   double scalar = x1 * q1 + x2 * q2;
            //   double norm = sqrt(x1*x1 + x2*x2) * sqrt(q1*q1 + q2*q2);
            //   Particles.getProp<F3>(l) = acos(scalar/norm);
            // }
        }
        if (v_cl.rank() == 0) {
            std::cout << "dt for the stepper is " << t-t_old << " Time Taken: "<<gt.getwct()
                      << "\n"
                      << "Time t = " << t << "\n";
        }
        if (ctr%wr_at==0 || ctr==wr_f){
            Particles.deleteGhost();
            Particles.write_frame("Polar",ctr,t);
            Particles.ghost_get<POLARIZATION>();
        }
        ctr++;

        dPol[x]=Pol[x]-Pol_old[x];
        dPol[y]=Pol[y]-Pol_old[y];

        //f6 = FranckEnergyDensity;

        std::vector<double> fes;
        int ctr2 = 0;
        for (int j =0; j<bulk.size();j++) {
            sum2 += Particles.getProp<FE>(j);
            anglesum += Particles.getProp<F3>(j);
            ctr2++;
        }
        sum2 = sum2/ctr2;
        anglesum = anglesum/ctr2;

        v_cl.max(sum2);
        v_cl.execute();
        std::cout << "ctr = "<< ctr << '\n';
        std::cout << "v_cl_rank = " << v_cl.rank() << '\n';
        std::cout << "FEsum_new = " << sum2 << '\n';
        std::cout << "angle = " << anglesum << '\n';
        bool minimum;
        for (int i=0; i<fes.size();i++){if(sum2<fes[i]){minimum=true;}}
        if (minimum && ctr >100 && fabs(anglesum_old-anglesum)<=steady_tol) {
            tt2.stop();
            if(v_cl.rank()==0)
                  {std::cout<<"Steady State Reached at angle = " << anglesum <<'\n';
                  std::cout << "The simulation took " << tt2.getcputime() << "(CPU) ------ " << tt2.getwct()
                  << "(Wall) Seconds.";}
            openfpm_finalize();
            exit(0);
        }

        // std::cout << "v_cl_rank = " << v_cl.rank() << '\n';
        // double MaxRateOfChange=0;
        // for (int j = 0; j < bulk.size(); j++) {
        //     auto p = bulk.get<0>(j);
        //     for (int i=0;i<2;i++){
        //         if(fabs((Particles.getProp<DPOL>(p)[i]))>MaxRateOfChange)
        //         {
        //             MaxRateOfChange=fabs(Particles.getProp<DPOL>(p)[i]);
        //         }
        //     }
        // }
        // v_cl.max(MaxRateOfChange);
        // v_cl.execute();
        // if(v_cl.rank()==0)
        // {std::cout<<"MaxRateOfChange: "<<MaxRateOfChange<<std::endl;
        // }
        // if(MaxRateOfChange<steady_tol && ctr>5)
        // {
        //     tt2.stop();
        //     if(v_cl.rank()==0)
        //     {std::cout<<"Steady State Reached."<<std::endl;
        //     std::cout << "The simulation took " << tt2.getcputime() << "(CPU) ------ " << tt2.getwct()
        //               << "(Wall) Seconds.";}
        //
        //     openfpm_finalize();
        //     exit(0);
        // }

        if (v_cl.rank()==0)
        {
            std::cout << "-----------------------------------------------------------------\n";
        }

        sum3 = sum2;
        fes.push_back(sum2);
        anglesum_old = anglesum;
        anglesum =0.0;
        sum2 = 0.0;
        Pol_old = Pol;
        dPol=0;
        t_old=t;
        state.data.get<0>()=Pol[x];
        state.data.get<1>()=Pol[y];

        gt.start();
    }
};

int main(int argc, char* argv[])
{
    {   openfpm_init(&argc,&argv);

        auto &v_cl = create_vcluster();

        tt2.start();
        size_t Gd = int(std::atof(argv[1]));
        double tf = std::atof(argv[2]);
        double dt = tf/std::atof(argv[3]);
        wr_f=int(std::atof(argv[3]));
        wr_at=int(std::atof(argv[4]));
        V_err_eps =1e-8; //chnge dependent of Gd
        //give dimensionless value for activity as 5th value to program
        zetadelmu = double(std::atof(argv[5]));
        //give dimensionless value for slpay/bend as 6th value to program
        dkK = double(std::atof(argv[6]));
        //std::cout << "dkK = "<<dkK << '\n';
        //give f3 as 7th value for program
        double phi = M_PI*double(std::atof(argv[7]));
        Kb = (dkK + 1.) * Ks;
        std::cout << "Kb = " << Kb << '\n';
        //calculate delmu from given parameter
        std::cout << "nu*gama*lambda= " << nu * gama * lambda << '\n';
        //delmu = -0.8;
        delmu = (zetadelmu * Ks)/(pow((Gd/2.), 2.) * (zeta + nu * gama * lambda));
        std::cout << "delmu = " << delmu << '\n';

        //std::cout << "Kb = "<< Kb << '\n';
        double boxsize = 10.0;
        const size_t sz[2] = {Gd, Gd};
        Box<2, double> box({0.0, 0.0}, {boxsize, boxsize});
        double Lx = box.getHigh(0),Ly = box.getHigh(1);
        size_t bc[2] = {NON_PERIODIC,NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0]),rCut = 3.9 * spacing;
        int ord = 2;
        Ghost<2, double> ghost(rCut);
        vector_dist_ws<2, double,Activegels> Particles(0, box, bc, ghost);
        Particles.setPropNames(PropNAMES);

        double x0=0.0, y0 = 0.0, x1=boxsize, y1=boxsize;
        double cd1=1;
        Point<2,double> Xpn,xpd1={5.,5.};

        //kernel
        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            auto key = it.get();
            double x = key.get(0) * spacing;
            double y = key.get(1) * spacing;
            //define cut off radius
            if(sqrt((5.0-x)*(5.0-x)+(5.0-y)*(5.0-y))<5.0-7*spacing/6.0)
            {
                Particles.add();
                Particles.getLastPos()[0] = x;
                Particles.getLastPos()[1] = y;
                Particles.getLastProp<NORMAL>()[0] = 0;
                Particles.getLastProp<NORMAL>()[1] = 0;
                Particles.getLastProp<PID>() = 0;
                Particles.getLastSubset(0);
            }
            ++it;
        }

        //define circle as grid for outer particles
        if (v_cl.rank()==0){
            int n_b=int(Gd)*5;
            double radius = 5.0 - 3*spacing/4.0;
            //double Golden_f3=M_PI * (3.0 - sqrt(5.0));
            double Golden_f3=2.0*M_PI/double(n_b);
                for(int i=1;i<=n_b;i++)
                {
                    double Golden_theta = Golden_f3 * i;
                    double x = 5.0+cos(Golden_theta) * radius;
                    double y = 5.0+sin(Golden_theta) * radius;
                    Particles.add();
                    Particles.getLastPos()[0] = x;
                    Particles.getLastPos()[1] = y;
                    Particles.getLastSubset(0);
                    Particles.getLastProp<NORMAL>()[0] = (x-5.0)/sqrt((x-5.0)*(x-5.0)+(y-5.0)*(y-5.0));
                    Particles.getLastProp<NORMAL>()[1] = (y-5.0)/sqrt((x-5.0)*(x-5.0)+(y-5.0)*(y-5.0));
                    Particles.getLastProp<PID>() = -1;
                    /*Particles.getLastProp<8>()[0] = 1.0 ;
                    Particles.getLastProp<8>()[1] = std::atan2(sqrt(x*x+y*y),z);
                    Particles.getLastProp<8>()[2] = std::atan2(y,x);*/
                }
        }
        Particles.map();

        size_t pctr=1;
        vector_dist_subset<2, double, Activegels> Particles_bulk(Particles,0);
        //auto f3 = getV<F3>(Particles);
        //Neumann Boundary Conditions: set angle for beginning: Neumann BC imposed by CorrVec.
        auto & Bulk = Particles_bulk.getIds();
        double max = 0.2;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, max);
        for (int j = 0; j < Bulk.size(); j++) {
            auto p = Bulk.get<0>(j);
            Xpn={Particles.getPos(p)[0],Particles.getPos(p)[1]};
            double dist=Xpn.distance(xpd1);
            dist = (dist == 0) ? 1 : dist;
            //compute normal
            double x1 = (5. - Xpn[0])/dist;
            double x2 = (5. - Xpn[1])/dist;
            //set angle of rotation
            //rotate normal by angle
            Particles.getProp<POLARIZATION>(p)[x] = cos(phi)*x1 - sin(phi)*x2;
            Particles.getProp<POLARIZATION>(p)[y] = sin(phi)*x1 + cos(phi)*x2;


            double v = (dis(gen) - (max/2.)) * M_PI;
            Particles_bulk.getProp<POLARIZATION>(p)[x] = cos(v)*Particles_bulk.getProp<POLARIZATION>(p)[x] - sin(v)*Particles_bulk.getProp<POLARIZATION>(p)[y];
            Particles_bulk.getProp<POLARIZATION>(p)[y] = sin(v)*Particles_bulk.getProp<POLARIZATION>(p)[x] + cos(v)*Particles_bulk.getProp<POLARIZATION>(p)[y];
            Particles.getProp<F3>(p) = phi+v;
            //std::cout << "phi = "<< phi << '\n';
            if (Particles.getProp<PID>(p)!=0){
                double x = Xpn[0];
                double y = Xpn[1];
                double theta=atan2(y-5.0,x-5.0);
                Particles.add();
                Particles.getLastPos()[0] = 5.0*(1.0+cos(theta));
                Particles.getLastPos()[1] = 5.0*(1.0+sin(theta));
                Particles.getLastSubset(1);
                Particles.getLastProp<0>()[0] = Particles.getProp<0>(p)[0];
                Particles.getLastProp<0>()[1] = Particles.getProp<0>(p)[1];
                Particles.getLastProp<F3>() = Particles.getProp<F3>(p);
                Particles.getLastProp<PID>() = pctr;
                Particles.getProp<PID>(p) = pctr;
                pctr++;
            }
        }
        // for (int l =0; l<boundary.size(); l++){
        //   Particles.getProp<F3>()
        // }
        //Particles.map();
        Particles.ghost_get<0,12,DELMU,NORMAL>();
        Particles.ghost_get<F3>();
        //Particles_bulk.update();
        //insert randomness to angle -> not already start with steady state
        vector_dist_subset<2, double, Activegels> Particles_boundary(Particles,1);
        auto & bulk = Particles_bulk.getIds();
        auto & boundary = Particles_boundary.getIds();
        //
        //
        // for (int l = 0; l < bulk.size(); l++){
        //   double v = (dis(gen) - (max/2.)) * M_PI;
        //   Particles_bulk.getProp<POLARIZATION>(l)[x] = cos(v)*Particles_bulk.getProp<POLARIZATION>(l)[x] - sin(v)*Particles_bulk.getProp<POLARIZATION>(l)[y];
        //   Particles_bulk.getProp<POLARIZATION>(l)[y] = sin(v)*Particles_bulk.getProp<POLARIZATION>(l)[x] + cos(v)*Particles_bulk.getProp<POLARIZATION>(l)[y];
        //   //Particles_boundary.getProp<F3>(l) = Particles_boundary.getProp<F3>(l) + v;
        //   // auto p = Bulk.get<0>(l);
        //   // if (Particles.getProp<PID>(p)!=0){
        //   //   Particles.getLastProp<F3>() = Particles.getProp<F3>(p);
        //   // }
        // }
        Particles.ghost_get<POLARIZATION, F3>(SKIP_LABELLING);


        //auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<POLARIZATION>(Particles);
        auto V = getV<VELOCITY>(Particles);
        //auto g = getV<NORMAL>(Particles);
        auto P = getV<PRESSURE>(Particles);
        //auto delmu = getV<DELMU>(Particles);
        auto dPol = getV<DPOL>(Particles);
        auto Pol_old = getV<POLD>(Particles);
        auto f3 = getV<F3>(Particles);

        //delmu = 0.5;
        //calculate delmu from given parameter
        //delmu = -70.0;//(zetadelmu * Ks)/(pow(Gd, 2.) * (zeta + nu * gama * lambda));
        //std::cout << "delmu = " << delmu << '\n';
        P = 0;V = 0,dPol=0,Pol_old=0;
        sum2 =0.;
        sum3 = 0.;
        anglesum = 0.0;
        anglesum_old =0.0;

        Particles.ghost_get<POLARIZATION,NORMAL,DELMU,DPOL, F3>(SKIP_LABELLING);


        // now link the partners of boundary and bulk
        //before PID for both particles at same time, bt still need to iterate, bc jusr to find partners
        //find or access single particles -> CorrVec
        CorrVec.clear();
        for(int i = 0; i < bulk.size(); i++)
        {
                auto p = bulk.get<0>(i); //0 for id by convention
                if(Particles.getProp<NORMAL>(p)[0]!=0 || Particles.getProp<NORMAL>(p)[1]!=0){
                    for(int j = 0; j < boundary.size(); j++) {
                        auto p_out = boundary.get<0>(j);
                        if(Particles.getProp<PID>(p_out)==Particles.getProp<PID>(p)){
                            CorrVec.add();
                            CorrVec.get<0>(CorrVec.size()-1)[0]=p_out;
                            CorrVec.get<0>(CorrVec.size()-1)[1]=p;
                            //std::cout<<p_out<<":"<<p<<std::endl;
                        }
                    }
                }
        }

        std::cout << "CORRVEC SIZE " << CorrVec.size() << "\n";


        auto P_bulk = getV<PRESSURE>(Particles_bulk);//Pressure only on inside
        auto Pol_bulk = getV<POLARIZATION>(Particles_bulk);
        auto dPol_bulk = getV<DPOL>(Particles_bulk);
        auto dV_bulk = getV<DV>(Particles_bulk);
        auto RHS_bulk = getV<VRHS>(Particles_bulk);
        auto div_bulk = getV<DIV>(Particles_bulk);


        // //insert randomness to angle -> not already start with steady state
        // double max = 0.2;
        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::uniform_real_distribution<> dis(0.0, max);
        //
        // for (int k = 0; k<boundary.size(); k++){
        //   double s = (dis(gen) - (max/2.)) * M_PI;
        //   Particles_boundary.getProp<POLARIZATION>(k)[x] = cos(s)*Particles_boundary.getProp<POLARIZATION>(k)[x] - sin(s)*Particles_boundary.getProp<POLARIZATION>(k)[y];
        //   Particles_boundary.getProp<POLARIZATION>(k)[y] = sin(s)*Particles_boundary.getProp<POLARIZATION>(k)[x] + cos(s)*Particles_boundary.getProp<POLARIZATION>(k)[y];
        //
        // }
        // for (int l = 0; l < bulk.size(); l++){
        //   double v = (dis(gen) - (max/2.)) * M_PI;
        //   Particles_bulk.getProp<POLARIZATION>(l)[x] = cos(v)*Particles_bulk.getProp<POLARIZATION>(l)[x] - sin(v)*Particles_bulk.getProp<POLARIZATION>(l)[y];
        //   Particles_bulk.getProp<POLARIZATION>(l)[y] = sin(v)*Particles_bulk.getProp<POLARIZATION>(l)[x] + cos(v)*Particles_bulk.getProp<POLARIZATION>(l)[y];
        //   //Particles_boundary.getProp<F3>(l) = Particles_boundary.getProp<F3>(l) + v;
        // }
        // //Particles.ghost_get<F3>();

        Particles.write("Init");

        Derivative_x Dx(Particles,ord,rCut,3.1,support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut,3.1,support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut,1.9,support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord, rCut,1.9,support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut,1.9,support_options::RADIUS);

        // boost::numeric::odeint::runge_kutta4< state_type_2d_ofp,double,state_type_2d_ofp,double,boost::numeric::odeint::vector_space_algebra_ofp> rk4;
       // boost::numeric::odeint::adams_bashforth_moulton<2,state_type_2d_ofp,double,state_type_2d_ofp,double,boost::numeric::odeint::vector_space_algebra_ofp> abm;
        boost::numeric::odeint::adaptive_adams_bashforth_moulton<2, state_type_2d_ofp,double,state_type_2d_ofp,double,boost::numeric::odeint::vector_space_algebra_ofp > abmA;

        vectorGlobal=(void *) &Particles;
        vectorGlobal_bulk=(void *) &Particles_bulk;
        vectorGlobal_boundary=(void *) &Particles_boundary;

        PolarEv<Derivative_x,Derivative_y,Derivative_xx,Derivative_xy,Derivative_yy> System(Dx,Dy,Dxx,Dxy,Dyy);
        CalcVelocity<Derivative_x,Derivative_y,Derivative_xx,Derivative_xy,Derivative_yy> CalcVelocityObserver(Dx,Dy,Dxx,Dxy,Dyy);

        state_type_2d_ofp tPol;
        tPol.data.get<0>()=Pol[x];
        tPol.data.get<1>()=Pol[y];

        eq_id vx, vy;
        vx.setId(0);
        vy.setId(1);
        timer tt;
        timer tt3;
        dPol = Pol;
        double V_err = 1, V_err_old;
        double tim=0;

        // intermediate time steps
        std::vector<double> inter_times;
        size_t steps;

        steps=boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled(1e-6,1e-6,abmA),System,tPol,tim,tf,dt,CalcVelocityObserver);
        //steps=boost::numeric::odeint::integrate_const(euler,System,tPol,tim,tf,dt,CalcVelocityObserver);

        std::cout << "Time steps: " << steps << std::endl;

        Pol_bulk[x]=tPol.data.get<0>();
        Pol_bulk[y]=tPol.data.get<1>();

        Particles.deleteGhost();
        Particles.write("Polar_Last");
        Dx.deallocate(Particles);
        Dy.deallocate(Particles);
        Dxy.deallocate(Particles);
        Dxx.deallocate(Particles);
        Dyy.deallocate(Particles);
        std::cout.precision(17);
        tt2.stop();
        if (v_cl.rank() == 0) {
            std::cout << "The simulation took " << tt2.getcputime() << "(CPU) ------ " << tt2.getwct()
                      << "(Wall) Seconds.";
        }
    }
    openfpm_finalize();
}
