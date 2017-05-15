/*
 * Adam.h
 *
 *  Created on: 2017/02/22
 *      Author: haga
 */

#ifndef ADAM_H_
#define ADAM_H_

#include "GradientDescent.h"

namespace Learn {

namespace adam {
constexpr double beta1 = 0.9;
constexpr double beta2 = 0.999;
constexpr double epsilon = 10e-8;
constexpr double alpha = 0.001;
}

class Adam : public GradientDescent{
private:
	unsigned int vecSize;
	Eigen::VectorXd mVec;
	Eigen::VectorXd vVec;
	unsigned int epoch;
public:
	Adam(unsigned int size);
	virtual ~Adam();

	void execute(Eigen::VectorXd& oldWeight, const Eigen::VectorXd& gVec) override ;
	void execute(Eigen::Vector3d& oldAns, const Eigen::Vector3d& gVec);
};

} /* namespace Learn */

#endif /* ADAM_H_ */
