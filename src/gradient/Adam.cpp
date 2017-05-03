/*
 * Adam.cpp
 *
 *  Created on: 2017/02/22
 *      Author: haga
 */

#include "Adam.h"

#include <cmath>

using namespace Eigen;

namespace Learn {

Adam::Adam(unsigned int size):
		vecSize(size),
		mVec(Eigen::VectorXd::Zero(size)),
		vVec(Eigen::VectorXd::Zero(size)),
		beta1(0.9),
		beta2(0.999),
		epsilon(10e-8),
		alpha(0.001),
		epoch(1){}

Adam::~Adam() = default;

void Adam::execute(Eigen::VectorXd& wVec, const Eigen::VectorXd& gVec){
	mVec = beta1*mVec + (1.0-beta1)*gVec;

	const VectorXd ggVec = gVec.array()*gVec.array();
	vVec = beta2*vVec + ((1.0-beta2)*(ggVec));

	const VectorXd _mVec = mVec/(1-std::pow(beta1, epoch));
	const VectorXd _vVec = vVec/(1-std::pow(beta2, epoch));

	//_mVec/sqrt(_vVec)+epsilon
	const VectorXd tmp = _mVec.array()/
			(_vVec.unaryExpr([](double e){return std::sqrt(e);}).array() + epsilon);

	//weightの更新
	wVec -= alpha*tmp;

	++epoch;
}

void Adam::execute(Eigen::Vector3d& wVec, const Eigen::Vector3d& gVec){
	mVec = beta1*mVec + (1.0-beta1)*gVec;

	const Vector3d ggVec = gVec.array()*gVec.array();
	vVec = beta2*vVec + ((1.0-beta2)*(ggVec));

	const Vector3d _mVec = mVec/(1-std::pow(beta1, epoch));
	const Vector3d _vVec = vVec/(1-std::pow(beta2, epoch));

	//_mVec/sqrt(_vVec)+epsilon
	const Vector3d tmp = _mVec.array()/
			(_vVec.unaryExpr([](double e){return std::sqrt(e);}).array() + epsilon);

	//weightの更新
	wVec -= alpha*tmp;

	++epoch;
}

} /* namespace Learn */
