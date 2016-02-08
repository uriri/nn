/*
 * LearningMethod.h
 *
 *  Created on: 2015/12/10
 *      Author: haga
 */

#ifndef LEARNINGMETHOD_H_
#define LEARNINGMETHOD_H_

#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Core>

namespace Learn {

template<class Vector>
Eigen::Matrix<typename Vector::value_type, Eigen::Dynamic, 1> STL2Vec(Vector& vector) {
	typedef typename Vector::value_type value_type;
	return Eigen::Map<Eigen::Matrix<value_type, Eigen::Dynamic, 1> >(&vector[0], vector.size(), 1);
}

constexpr double  alpha = 5.0;//シグモイド関数のゲイン

class LearningMethod {
private:
	std::ifstream m_ifs;
	unsigned int m_weightSize;

	double mu; //metaLearningRate
	double lamda;

	Eigen::VectorXd pVec;
	Eigen::VectorXd vVec;

public:
	LearningMethod(const int weightSize, double learningRate, double meta, double lam);
	virtual ~LearningMethod();

	//損失関数、中身はシグモイド
	double lossFunc(double arg){
		return ( 1.0/(1.0+exp(-alpha*arg)) );
//		return std::max(arg, 0.0);
	}

	double d_LossFunc(double arg){
		return ( alpha*lossFunc(arg)*(1.0-lossFunc(arg)) );
//		return arg>0.0?1.0:0.0;
	}

	double dd_LossFunc(double arg){
		return ( lossFunc(arg)*(1.0-lossFunc(arg))*(1.0-2*lossFunc(arg)) );
	}


#if 0
	//誤差逆伝播法による重みの修正
	template<int outSize>
	void backPropagation(double (&teach)[outSize]) {

		for (int i = 0; i < m_weightVecH2O.size(); ++i) {
			for (int j = 0; j < m_weightVecH2O[i].size(); ++j) {
				double delta = m_learningRate * (teach[i] - sigmoid(m_output[i]))
						* diffeSigmoid(m_output[i]) * m_hiddenLayer[j];
				m_weightVecH2O[i][j] += delta;
			}
		}

		for (int i = 0; i < m_hiddenLayer.size() - 1; ++i) {
			double tmp1 = 0.0;
			double tmp2 = 0.0;
			for (int k = 0; k < m_output.size(); ++k) {
				tmp1 += m_weightVecH2O[k][i] * (teach[k] - sigmoid(m_output[k]))
						* diffeSigmoid(m_output[k]);
			}
			tmp2 = diffeSigmoid(m_hiddenLayer[i]) * tmp1;
			for (int j = 0; j < m_input.size(); ++j) {
				double delta = m_learningRate * tmp2 * m_input[j];
				m_weightVecI2H[i][j] += delta;
			}
		}
	}
#endif

	//Stochastic Meta-Descentによる重み更新
	void SMD(std::vector<double>& oldWeight, const Eigen::VectorXd& loss);
	void SMD(Eigen::Vector3d& oldAns, const Eigen::Vector3d& gVec);

	double getLearningRateNorm() const {
		return pVec.norm();
	}

	Eigen::VectorXd getLearningRate() const {
		return pVec;
	}

};

} /* namespace NN */

#endif /* LEARNINGMETHOD_H_ */
