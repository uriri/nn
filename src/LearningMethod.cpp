/*
 * LearningMethod.cpp
 *
 *  Created on: 2015/12/10
 *      Author: haga
 */

#include "LearningMethod.h"

#include <iostream>

using namespace std;
using namespace Eigen;

namespace Learn {

LearningMethod::LearningMethod(const int weightSize, double learningRate) {

	m_weightSize = weightSize;
	pVec = VectorXd::Ones(m_weightSize);
	vVec = VectorXd::Zero(m_weightSize);

	pVec *= learningRate;//学習率をベクトルに拡張

	mu = 1e-6; //meta-LearningRate
	lamda = 0;
}

LearningMethod::~LearningMethod() {
	// TODO Auto-generated destructor stub
}

//Stochastic Meta-Descent
void LearningMethod::SMD(std::vector<double>& oldWeight, const Eigen::VectorXd& loss) {

	auto diag = [&](VectorXd arg) {
		MatrixXd unitMat(m_weightSize, m_weightSize);
		unitMat = arg.asDiagonal();
		return unitMat;
	};

	auto oneHalfOver = [&](VectorXd arg){
		for(int i=0; i<m_weightSize; ++i){
			arg(i) = arg(i)<0.5?0.5:arg(i);
		}
		return arg;
	};

//	cout << "oldWeight" << endl;
//	for(auto ol : oldWeight){
//		cout << "  " << ol << endl;
//	}

	VectorXd nextWeight(m_weightSize);
	nextWeight = STL2Vec(oldWeight);

	//local learning rateの更新
	VectorXd tmp = oneHalfOver( VectorXd::Ones(m_weightSize)+(mu*diag(vVec)*loss) );
	pVec = diag(pVec)*tmp;

	//weightの更新
	nextWeight -= diag(pVec)*loss;

	//auxiliary vの更新
	vVec = lamda*vVec + diag(pVec)*( loss-lamda*(loss*(loss.dot(vVec))) );

	for(int i=0; i<m_weightSize; ++i){
		oldWeight[i] = nextWeight[i];
	}
}

//3次元用
void LearningMethod::SMD(Eigen::Vector3d& oldAns, const Eigen::Vector3d& gVec) {

	auto diag = [&](Vector3d arg) {
		Matrix3d unitMat;
		unitMat = arg.asDiagonal();
		return unitMat;
	};

	auto oneHalfOver = [&](Vector3d arg){
		for(int i=0; i<3; ++i){
			if(arg[i] < 0.5)
				arg[i] = 0.5;
		}
		return arg;
	};

	Vector3d nextAns = oldAns;

	//local learning rateの更新
	pVec = diag(pVec)*oneHalfOver( Vector3d::Ones()+(mu*diag(vVec)*gVec) );

	//weightの更新
	nextAns -= diag(pVec)*gVec;

	//auxiliary vの更新
	vVec = lamda*vVec + diag(pVec)*( gVec - lamda*(gVec*(gVec.dot(vVec))) );

	oldAns = nextAns;
}

template<int Size>
void printArray(int (&arg)[Size]) {
	for (int i = 0; i < Size; ++i) {
		cout << arg[i] << " ";
	}
	cout << endl;
}

} /* namespace NN */
