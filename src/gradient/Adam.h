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

class Adam : public GradientDescent{
private:
	unsigned int vecSize;
	Eigen::VectorXd mVec;
	Eigen::VectorXd vVec;
	double beta1;
	double beta2;
	double epsilon;
	double alpha;
	unsigned int epoch;
public:
	Adam(unsigned int size);
	virtual ~Adam();

	void execute(Eigen::VectorXd& oldWeight, const Eigen::VectorXd& gVec) override ;
	void execute(Eigen::Vector3d& oldAns, const Eigen::Vector3d& gVec);
};

} /* namespace Learn */

#endif /* ADAM_H_ */
