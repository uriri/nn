/*
 * SMD.h
 *
 *  Created on: 2015/12/10
 *      Author: haga
 */

#ifndef SMD_H_
#define SMD_H_

#include "GradientDescent.h"

namespace Learn {

class SMD : public  GradientDescent{
private:
	unsigned int m_vecSize;

	double mu; //metaLearningRate
	double lamda;

	Eigen::VectorXd pVec;
	Eigen::VectorXd vVec;

public:

	SMD(const int weightSize, double learningRate, double meta, double lam);
	virtual ~SMD();

	//Stochastic Meta-Descentによる重み更新
	void execute(Eigen::VectorXd& oldWeight, const Eigen::VectorXd& loss) override ;
	void execute(Eigen::Vector3d& oldAns, const Eigen::Vector3d& gVec);

	double getLearningRateNorm() const {
		return pVec.norm();
	}

	Eigen::VectorXd getLearningRate() const {
		return pVec;
	}

};

} /* namespace NN */

#endif /* SMD_H_ */
