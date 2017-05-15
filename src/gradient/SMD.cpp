/*
 * SMD.cpp
 *
 *  Created on: 2015/12/10
 *      Author: haga
 */

#include "SMD.h"

#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;

namespace Learn {

SMD::SMD(const int weightSize, double learningRate, double meta, double lam):
		m_vecSize(weightSize), mu(meta), lamda(lam),
		pVec(VectorXd::Constant(weightSize, learningRate)), //学習率をベクトルに拡張
		vVec(VectorXd::Zero(weightSize)) {}

SMD::~SMD() = default;

//Stochastic Meta-Descent
void SMD::execute(Eigen::VectorXd& oldWeight, const Eigen::VectorXd& loss) {

//	cout << "oldWeight" << endl;
//	for(auto ol : oldWeight){
//		cout << "  " << ol << endl;
//	}

	//local learning rateの更新
	pVec = pVec.asDiagonal()*
			(VectorXd::Ones(m_vecSize) + (mu*vVec.asDiagonal()*loss)).unaryExpr(
					[](double e){return max(0.5, e);}
	);

	//weightの更新
	VectorXd nextWeight(m_vecSize);
	nextWeight = ( oldWeight - (pVec.asDiagonal()*loss) );

	//auxiliary vの更新
	vVec = lamda*vVec + pVec.asDiagonal()*( loss-(lamda*(loss*loss.dot(vVec))) );

//	if(nextWeight == STL2Vec(oldWeight)){
//		cout << "not update" << endl;
//	}

	Map<VectorXd>(&oldWeight[0], m_vecSize) = nextWeight;
}

//3次元用
void SMD::execute(Eigen::Vector3d& oldAns, const Eigen::Vector3d& gVec) {

	//local learning rateの更新
	pVec = pVec.asDiagonal()*
			(Vector3d::Ones()+(mu*vVec.asDiagonal()*gVec)).unaryExpr(
					[](double e){return max(0.5, e);}
	);

	//weightの更新
	Vector3d nextAns =oldAns - pVec.asDiagonal()*gVec;

	//auxiliary vの更新
	vVec = lamda*vVec + pVec.asDiagonal()*( gVec - lamda*(gVec*gVec.transpose()*vVec) );

	oldAns = nextAns;
};

template<int Size>
void printArray(int (&arg)[Size]) {
	for (int i = 0; i < Size; ++i) {
		cout << arg[i] << " ";
	}
	cout << endl;
}

} /* namespace NN */
