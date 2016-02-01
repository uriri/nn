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

LearningMethod::LearningMethod(const int weightSize, double learningRate, double meta, double lam) {

	m_weightSize = weightSize;
	pVec = VectorXd::Ones(m_weightSize);
	vVec = VectorXd::Zero(m_weightSize);

	pVec *= learningRate;//学習率をベクトルに拡張

	mu = meta; //meta-LearningRate
	lamda = lam;
}

LearningMethod::~LearningMethod() {
	// TODO Auto-generated destructor stub
}

//Stochastic Meta-Descent
void LearningMethod::SMD(std::vector<double>& oldWeight, const Eigen::VectorXd& loss) {

	auto oneHalfOver = [&](VectorXd arg){
		for(int i=0; i<m_weightSize; ++i){
			arg[i] = std::max(0.5, arg[i]);
		}
		return arg;
	};

//	cout << "oldWeight" << endl;
//	for(auto ol : oldWeight){
//		cout << "  " << ol << endl;
//	}


	//local learning rateの更新
	VectorXd tmp = oneHalfOver( VectorXd::Ones(m_weightSize)+(mu*vVec.asDiagonal()*loss) );
	pVec = pVec.asDiagonal()*tmp;

	//weightの更新
	VectorXd nextWeight(m_weightSize);
	nextWeight = STL2Vec(oldWeight) - pVec.asDiagonal()*loss;

	//auxiliary vの更新
	vVec = lamda*vVec + pVec.asDiagonal()*( loss-lamda*(loss*loss.transpose()*vVec) );

	if(nextWeight == STL2Vec(oldWeight)){
		cout << "not update" << endl;
	}

	for(int i=0; i<m_weightSize; ++i){
		oldWeight[i] = nextWeight[i];
	}
}

//3次元用
void LearningMethod::SMD(Eigen::Vector3d& oldAns, const Eigen::Vector3d& gVec) {

	auto oneHalfOver = [&](Vector3d arg){
		for(int i=0; i<3; ++i){
			if(arg[i] < 0.5)
				arg[i] = 0.5;
		}
		return arg;
	};

	Vector3d nextAns = oldAns;

	//local learning rateの更新
	pVec = pVec.asDiagonal()*oneHalfOver( Vector3d::Ones()+(mu*vVec.asDiagonal()*gVec) );

	//weightの更新
	nextAns -= pVec.asDiagonal()*gVec;

	//auxiliary vの更新
	vVec = lamda*vVec + pVec.asDiagonal()*( gVec - lamda*(gVec*gVec.transpose()*vVec) );

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
