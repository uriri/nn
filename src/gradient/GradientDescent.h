/*
 * GradiendDescent.h
 *
 *  Created on: 2017/03/08
 *      Author: haga
 */

#ifndef GRADIENT_GRADIENTDESCENT_H_
#define GRADIENT_GRADIENTDESCENT_H_

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

class GradientDescent {
public:
	virtual ~GradientDescent() = default;

	virtual void execute(Eigen::VectorXd& oldWeight, const Eigen::VectorXd& gVec) = 0;
};

} /* namespace Learn */

#endif /* GRADIENT_GRADIENTDESCENT_H_ */
