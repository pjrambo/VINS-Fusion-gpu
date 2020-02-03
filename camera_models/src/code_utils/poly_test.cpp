#include <iostream>
#include <eigen3/Eigen/Core>

#include <code_utils/math_utils/Polynomial.h>
#include <code_utils/eigen_utils.h>

#include <code_utils/sys_utils.h>

using namespace std;
using namespace Eigen;

void fit_test()
{
  eigen_utils::Vector xx(101);
  eigen_utils::Vector out(101);

  xx<<
  0 ,0.0261799 ,0.0523599 ,0.0785398  , 0.10472  ,  0.1309 ,  0.15708  , 0.18326   ,
      0.20944 , 0.235619 , 0.261799  ,0.287979  ,0.314159 , 0.340339  ,0.366519   ,
     0.392699 , 0.418879 , 0.445059  ,0.471239  ,0.497419 , 0.523599  ,0.549779   ,
     0.575959 , 0.602139 , 0.628319  ,0.654498  ,0.680678 , 0.706858  ,0.733038   ,
      0.759218,  0.785398,  0.811578 , 0.837758 , 0.863938,  0.890118 , 0.916298  ,
      0.942478,  0.968658,  0.994838 ,  1.02102 ,   1.0472,   1.07338 ,  1.09956  ,
      1.12574 ,  1.15192 ,   1.1781  , 1.20428  , 1.23046 ,  1.25664  , 1.28282   ,
        1.309 ,  1.33518 ,  1.36136  , 1.38754  , 1.41372 ,   1.4399  , 1.46608   ,
      1.49226 ,  1.51844 ,  1.54462  ,  1.5708  , 1.59698 ,  1.62316  , 1.64934   ,
      1.67552 ,   1.7017 ,  1.72788  , 1.75406  , 1.78024 ,  1.80642  ,  1.8326   ,
      1.85878 ,  1.88496 ,  1.91114  , 1.93732  ,  1.9635 ,  1.98968  , 2.01586   ,
      2.04204 ,  2.06822 ,   2.0944  , 2.12058  , 2.14675 ,  2.17293  , 2.19911   ,
      2.22529 ,  2.25147 ,  2.27765  , 2.30383  , 2.33001 ,  2.35619  , 2.38237   ,
      2.40855 ,  2.43473 ,  2.46091  , 2.48709  , 2.51327 ,  2.53945  , 2.56563   ,
      2.59181 ,  2.61799 ;

 out<< 0,
  0.0261771 ,0.0523467, 0.0785066,  0.104654 , 0.130789 , 0.156907,  0.183008 , 0.209091,  0.235154 , 0.261196 ,
  0.287216  ,0.313212 , 0.339184 , 0.365131  ,0.391052  ,0.416945 ,  0.44281  ,0.468646 , 0.494452  ,0.520226  ,
  0.545968  ,0.571676 , 0.597349 , 0.622986  ,0.648584  ,0.674144 , 0.699662  ,0.725137 , 0.750567  ,0.775951  ,
  0.801285  ,0.826568 , 0.851797 , 0.876968  , 0.90208  ,0.927128 ,  0.95211  ,0.977021 ,  1.00186  , 1.02661  ,
  1.05129   ,1.07587  , 1.10036  , 1.12475   ,1.14903   ,1.17319  , 1.19724   ,1.22115  , 1.24492   ,1.26855   ,
  1.29201   , 1.3153  , 1.33841  , 1.36132   ,1.38402   , 1.4065  , 1.42873   , 1.4507  ,  1.4724   ,1.49379   ,
  1.51486   ,1.53558  , 1.55594  , 1.57589   ,1.59541   ,1.61448  , 1.63305   ,1.65109  , 1.66857   ,1.68543   ,
  1.70165   ,1.71716  , 1.73193  ,  1.7459   ,  1.759   ,1.77119  ,  1.7824   ,1.79256  , 1.80159   ,1.80943   ,
  1.81599   ,1.82119  , 1.82494  , 1.82713   ,1.82767   ,1.82646  , 1.82338   ,1.81831  , 1.81112   ,1.80168   ,
  1.78986   ,1.77551  , 1.75847  , 1.73857   ,1.71566   ,1.68955  , 1.66005   ,1.62696  , 1.59008   ,1.54919   ;


  math_utils::PolynomialFit polyfit(24,out, xx);
  math_utils::Polynomial poly =  polyfit.getCoeff();
  std::cout << "polyfit :"<<endl <<poly.getPolyCoeff().transpose() <<std::endl;

  double dd = 0.02;
  for(int i =0;i<100;++i)
  {
    std::cout<<dd*i<<" "<<poly.getValue(dd*i)<<std::endl;
  }

}

void test_poly()
{
  eigen_utils::Vector coeff(6);
  coeff << 1,2,3,4,5,6;

  std::cout << "coeff :"<<endl << coeff.transpose() <<std::endl;

  math_utils::Polynomial poly(5);
  poly.setPolyCoeff(coeff);
  std::cout << "coeff :"<<endl << poly.getPolyCoeff().transpose() <<std::endl;

  for(int i = 0;i<10;i++)
  {
    std::cout << "x: "<< i << " y : " <<poly.getValue(i) <<std::endl;
  }

  eigen_utils::Vector xx = coeff;
  eigen_utils::Vector out = poly.getValue(xx);
  std::cout << "out :"<<endl << out.transpose() <<std::endl;

  std::cout << ":--PolynomialFit--------------------------------------:" << std::endl;
  math_utils::PolynomialFit polyfit(5,xx,out);
  std::cout << "polyfit :"<<endl << polyfit.getCoeff().transpose() <<std::endl;

  math_utils::Polynomial polyfited;
  polyfited = polyfit.getPolynomial();
  std::cout << "polyfited :"<<endl << polyfited.getPolyCoeff().transpose() <<std::endl;

  math_utils::Polynomial* polyfited2 = new math_utils::Polynomial();
  *polyfited2 = polyfit.getPolynomial();
  std::cout << "polyfited2 :"<<endl << polyfited2->getPolyCoeff().transpose() <<std::endl;

  std::cout << ":----------------------------------------:" << std::endl;
  eigen_utils::Vector realroot = poly.getRealRoot(0.0);
  std::cout << "  Roots :"<<endl << realroot.transpose() <<std::endl;

  std::cout << ":----------------------------------------:" << std::endl;
  //  eigen_utils::Vector  polyn = eigen_utils::SwapSequence(coeff);
  eigen_utils::Vector  polyn(4);
  polyn<<-5,-1,5,1;
  std::cout << "polyn :"<<endl << polyn.transpose() <<std::endl;
  math_utils::Polynomial poly2(polyn);
  eigen_utils::Vector realroot2 = poly2.getRealRoot(0.0);
  std::cout << "  Roots :"<<endl << realroot2.transpose() <<std::endl;

  std::cout << ":----------------------------------------:" << std::endl;
  eigen_utils::Vector  poly3(4);
  poly3<< -5,1,0,0;
  std::cout << "polyn :"<<endl << poly3.transpose() <<std::endl;
  math_utils::Polynomial polyn3(poly3);
  double realroot3 = polyn3.getOneRealRoot(0.0,-100,100);
  std::cout << " One Roots :"<<endl << realroot3 <<std::endl;

  std::cout << ":----------------------------------------:" << std::endl;
  math_utils::Polynomial polyn4;
  polyn4 = polyn3;
  std::cout << "polyn3 :"<<endl << polyn3.getPolyCoeff().transpose() <<std::endl;
  std::cout << "polyn4 :"<<endl << polyn4.getPolyCoeff().transpose() <<std::endl;
  std::cout << "polyn4 :"<<endl << polyn4.toString() <<std::endl;


}

int main()
{
  eigen_utils::Vector vec(3);
  std::cout << "size :"<<endl << vec.size() <<std::endl;
  vec << 1,2,1;
  vec = eigen_utils::pushback(vec,22);
  std::cout << "vec :"<<endl << vec.transpose() <<std::endl;

  test_poly();
  std::cout<< "    constant?"  << std::endl;;

//  double q_in[4] = {1,2,3,4};
//  double q_out[4] ;

  sys_utils::PrintWarning("warning");
  sys_utils::PrintError("PrintError");
  sys_utils::PrintInfo("PrintInfo");

  std::cout << ":----------------------------------------:" << std::endl;
  fit_test();

  return 0;
}
