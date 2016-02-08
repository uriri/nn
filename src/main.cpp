/*
 * main.cpp
 *
 *  Created on: 2015/12/03
 *      Author: haga
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <limits>

#include "NeuralNetwork.h"
#include "LearningMethod.h"

//#define DEBUG
//#define DEBUGL
//#define DEBUGT

#define LEARNING

#ifdef DEBUG
#define DOUT std::cout
#else
#define DOUT 0&&std::cout
#endif

#ifdef DEBUGL
#define DOUTL std::cout
#else
#define DOUTL 0&&std::cout
#endif

#ifdef DEBUGT
#define DOUTT std::cout
#else
#define DOUTT 0&&std::cout
#endif

template<int size>
void printArray(int (&arg)[size]) {
	using namespace std;
	for (int i = 0; i < size; ++i)
		DOUT << arg[i] << " ";
	DOUT << endl;
}

template<int Size>
void connectArray(int (&ans)[Size * 2], int (&a)[Size], int (&b)[Size]) {
	for (int i = 0; i < Size; ++i) {
		ans[i] = a[i];
	}
	for (int j = 0; j < Size; ++j) {
		ans[j + Size] = b[j];
	}
}

int main() {
	using namespace std;
	using namespace Eigen;
	using namespace NN;
	using namespace Learn;

	constexpr int inSize = 28;
	constexpr int hideSize = 20;
	constexpr int wSize = ((inSize + 1) * hideSize) + (hideSize + 1);

	constexpr double lamda = 0.0;

	constexpr double learningRate = 0.1;
	constexpr double mu = 0.05; //meta-LearningRate

	constexpr int learningTimes = 5000;

	//群平均の分母
	constexpr int geneNum = 10;

//	double learningRate;
//	cin >> learningRate;

	ifstream learnFile, testFile;

	NeuralNetwork neuNet(inSize, hideSize);
	LearningMethod lm(wSize, learningRate, mu, lamda);

//	double lr = learningRate;

#if 1

	int myHandRank[14];
	int bestHandRank[14], otherHandRank[14];
	int otherMvQty;
	double minLossValue;

	string learnFileName = "learn_500.dat";
	string testFileName = "test_500.dat";

	ostringstream ostrs;
	ostrs << "output/weight_" << hideSize << ".dat";
	const auto& outFile = ostrs.str();

//	cout << "inout" << endl;
//	for(auto ww : neuNet.getWeightOneDim()) {
//		cout << ww << " ";
//	}
//	cout << endl;

#ifdef LEARNING
	ofstream plot;

	string plotFile;
	bool isExist = true;
	unsigned int num = 0;
	do{
		ostringstream os;
		os << "plot" << num << ".csv";
		ifstream check(os.str().c_str());
		if(check.fail()){
			plotFile = os.str();
			isExist = false;
		}
		++num;
	}while(isExist);
	cout << plotFile << endl;
	plot.open(plotFile.c_str());

	learnFile.open(learnFileName.c_str(), ios::binary);
	testFile.open(testFileName.c_str(), ios::binary);

	if (learnFile.fail()) {
		cerr << "learn cannot open" << endl;
		return -1;
	}

	if(testFile.fail()) {
		cerr << "test cannot open " << endl;
		return -1;
	}

	chrono::system_clock::time_point start, end;
	start = chrono::system_clock::now();

	unsigned int notChangeCount = 0;
	double beforeLoss = numeric_limits<double>::max();
	double aveMatch = 0.0;

	bool isEnd = false;

	for (unsigned int itr = 1; itr <= learningTimes; ++itr) {

		VectorXd lossGrad = VectorXd::Zero(wSize);
		double lossValue;
		double totalLoss = 0.0;

		//ファイルの最後までループ
		while (!learnFile.eof()) {
			learnFile.read(reinterpret_cast<char*>(&otherMvQty), sizeof(otherMvQty));
			DOUT << "MvQty " << otherMvQty << endl;

			int input[28]; //自分の手札 + 相手の手札
			vector<double> bestGrad(wSize), otherGrad(wSize); //NNの勾配

			learnFile.read(reinterpret_cast<char*>(myHandRank), sizeof(myHandRank));
			learnFile.read(reinterpret_cast<char*>(bestHandRank), sizeof(bestHandRank));

			connectArray(input, myHandRank, bestHandRank);
			DOUT << " +best " << endl;
			printArray(input);

			neuNet.calOutPut(input);
			double bestValue = neuNet.getOutPut();
			bestGrad = neuNet.getGrad();

			DOUTL << "bestValue  " << bestValue << endl;

			for (int j = 0; j < otherMvQty; ++j) {
				learnFile.read(reinterpret_cast<char*>(otherHandRank), sizeof(otherHandRank));

				connectArray(input, myHandRank, otherHandRank);
				DOUT << " +other" << j << " " << endl;
				printArray(input);
				neuNet.calOutPut(input);
				double otherValue = neuNet.getOutPut();
				otherGrad = neuNet.getGrad();

				DOUTL << "otherValue " << otherValue << endl;

				//勾配をここで求めて総和を計算
				lossValue = otherValue - bestValue;
				DOUTL << "loss       " << lm.lossFunc(lossValue) << "(" << lossValue << ") " << lm.d_LossFunc(lossValue) << endl;
				totalLoss += lm.lossFunc(lossValue);
				lossGrad += ( lm.d_LossFunc(lossValue)*(STL2Vec(otherGrad) - STL2Vec(bestGrad)) );
			}
			DOUTL << "totalLoss  " << totalLoss << endl;
			DOUTL << endl;
		} //File end

		cout << itr << " : " << beforeLoss << "->" << totalLoss << " (" << totalLoss - beforeLoss << ")";
//		cout << ", " << lr;
		cout << endl;

		if(fabs(beforeLoss - totalLoss) > 1e-15){
			beforeLoss = totalLoss;
			notChangeCount = 0;
		} else {
			++notChangeCount;
			cout << "not change " << notChangeCount << endl;
		}

		if(notChangeCount >= 10){
			cout << "stop learn" << endl;
			break;
		}

		if( fabs(totalLoss) < 1e-15 ){
			cout << "Optimized totalLoss" << endl;
			minLossValue = totalLoss;
			break;
		}

#if 0
		if(i==0){
			minLossValue = totalLoss;
		} else {
//			cout << "min " << minLossValue << endl;
			if(totalLoss <= minLossValue){
				minLossValue = totalLoss;

				auto aftWeight = neuNet.getWeightOneDim();
				ostringstream out;
				out << "output/weight_" << hideSize << "_" << totalLoss << ".dat";

				ofstream ofs(out.str(), ios::binary | ios::trunc);
				ofs.write(reinterpret_cast<const char*>(&wSize), sizeof(wSize));
				ofs.write(reinterpret_cast<char*>(&aftWeight[0]), wSize * sizeof(double));
				ofs.close();
			}
		}
#endif

		auto wVec = neuNet.getWeightOneDim();

		lm.SMD(wVec, lossGrad);
		neuNet.setWeight(wVec);
//		cout << lm.getLearningRate().transpose() << endl;

//		VectorXd next = STL2Vec(wVec) - lr*lossGrad;
//		for(int i=0; i<wSize; ++i){
//			wVec[i] = next[i];
//		}
//		neuNet.setWeight(wVec);
//		lr *= mu;

		/*
		 * test
		 */

//		cout << "--- test ---" << endl;

		int bestCount = 0;
		int count = 0;

		//ファイルの最後までループ
		while (!testFile.eof()) {
			testFile.read(reinterpret_cast<char*>(&otherMvQty), sizeof(otherMvQty));
			DOUTT << "MvQty " << otherMvQty << endl;

			++count;
			int input[28]; //自分の手札 + 相手の手札
			bool isBest = true;

			testFile.read(reinterpret_cast<char*>(myHandRank), sizeof(myHandRank));
			testFile.read(reinterpret_cast<char*>(bestHandRank), sizeof(bestHandRank));

			connectArray(input, myHandRank, bestHandRank);
			neuNet.calOutPut(input);
			const double bestValue = neuNet.getOutPut();
			DOUTT << "bestValue  " << bestValue << endl;

			for (int j = 0; j < otherMvQty; ++j) {
				testFile.read(reinterpret_cast<char*>(otherHandRank), sizeof(otherHandRank));

				connectArray(input, myHandRank, otherHandRank);
				neuNet.calOutPut(input);
				const double otherValue = neuNet.getOutPut();
				DOUTT << "otherValue " << otherValue << endl;
				if( otherValue >= bestValue ){
					//1つでもダメだったらbreakでやめる <- バカ
					//for文全部まわんなくてファイル読み込みがずれる
					isBest = false;
				}
			}

			if(isBest){
				DOUTT << "ok" << endl;
				++bestCount;
			}
			DOUTT << endl;

		}//end testFile


		//汎化テストの一致率
		double match = (static_cast<double>(bestCount) / static_cast<double>(count))*100;

		plot << itr << "," << totalLoss << "," << match;

		//群平均
		aveMatch += match;

		if(itr%geneNum != 0){
			plot << ",=NA()";
		} else {
			plot << "," << aveMatch/geneNum;
			aveMatch = 0.0;
		}
		plot  << "," << lm.getLearningRateNorm();
		if(itr == 1){
			plot << ",," << "rate|meta(" << learningRate << "|" << mu << ")," << neuNet.getSeed();
		}
		plot << endl;

		testFile.clear();
		testFile.seekg(0, ios::beg);

		learnFile.clear();
		learnFile.seekg(0, ios::beg);

		if(itr==learningTimes)
			minLossValue = totalLoss;

	} //learning loop

	auto aftWeight = neuNet.getWeightOneDim();
	ostringstream out;
	out << "output/weight_" << hideSize << "_" << minLossValue << ".dat";

	ofstream ofs(out.str(), ios::binary | ios::trunc);
	ofs.write(reinterpret_cast<const char*>(&wSize), sizeof(wSize));
	ofs.write(reinterpret_cast<char*>(&aftWeight[0]), wSize*sizeof(double));
	ofs.close();

	learnFile.close();
	testFile.close();

	end = chrono::system_clock::now();

	double elapsed = chrono::duration_cast < chrono::seconds > (end - start).count();
	cout << endl << elapsed << "sec" << endl;

//	cout << "output" << endl;
//	for(auto ww : aftWeight) {
//		cout << ww << " ";
//	}
//	cout << endl;

#else

	testFile.open("learn_500.dat", ios::binary);

	if (testFile.fail()) {
		cerr << "test cannot open" << endl;
		return -1;
	}

	//損失関数の総和だけ入力すれば勝手にファイル名を生成してくれる
	string value;
	cout << "testFile lossValue " << endl;
	cin >> value;

	ostringstream out;
	out << "output/weight_" << hideSize << "_" << value << ".dat";

	cout << out.str() << "\n" << endl;
	neuNet.readWeightFile(out.str(), wSize);

//	auto aftw = neuNet.getWeightOneDim();
//	cout << "read aft weight" << endl;
//	for (auto w : aftw) {
//		cout << w << " ";
//	}
//	cout << endl;

	int bestCount = 0;
	int count = 0;

	//ファイルの最後までループ
	while (!testFile.eof()) {
		testFile.read(reinterpret_cast<char*>(&otherMvQty), sizeof(otherMvQty));
		DOUTT << "MvQty " << otherMvQty << endl;

		++count;
		int input[28]; //自分の手札 + 相手の手札
		bool isBest = true;

		testFile.read(reinterpret_cast<char*>(myHandRank), sizeof(myHandRank));
		testFile.read(reinterpret_cast<char*>(bestHandRank), sizeof(bestHandRank));

		connectArray(input, myHandRank, bestHandRank);
		neuNet.calOutPut(input);
		const double bestValue = neuNet.getOutPut();
		DOUTT << "bestValue  " << bestValue << endl;

		for (int j = 0; j < otherMvQty; ++j) {
			testFile.read(reinterpret_cast<char*>(otherHandRank), sizeof(otherHandRank));

			connectArray(input, myHandRank, otherHandRank);
			neuNet.calOutPut(input);
			const double otherValue = neuNet.getOutPut();
			DOUTT << "otherValue " << otherValue << endl;
			if( otherValue >= bestValue ){
				//1つでもダメだったらbreakでやめる <- バカがやることなのでやめましょう
				//readが呼ばれなくなります
				isBest = false;
			}
		}

		if(isBest){
			DOUTT << "ok" << endl;
			++bestCount;
		}
		DOUTT << endl;

	}//end testFile

	cout << "best/count -> " << bestCount << "/" << count << endl;
	const auto match = (static_cast<double>(bestCount) / static_cast<double>(count)) * 100;
	cout << match << endl;

	testFile.close();

	/*
	auto fileName = [](int hide, int index) {
		ostringstream ostrs;
		ostrs << "output/weight_" << hide;
		if(index >= 0) {
			ostrs << "_" << index;
		}
		ostrs << ".dat";
		return ostrs.str();
	};

	string bestWeight = fileName(hideSize, maxIndex);
	string bestOut = fileName(hideSize, -1);

	ifstream best(bestWeight.c_str(), ios::binary);
	ofstream out(bestOut, ios::binary);

	vector<double> ww(wSize);
	int ss = wSize;
	if (!best.fail()) {
		int readSize;
		best.read(reinterpret_cast<char*>(&readSize), sizeof(readSize));
		best.read(reinterpret_cast<char*>(&ww[0]), readSize * sizeof(double));
	} else {
		cerr << "cannot open" << endl;
	}
	best.close();

//	for(int i=0; i<ss; ++i){
//		cout << ww[i] << " ";
//	}cout << endl;

	out.write(reinterpret_cast<char*>(&ss), sizeof(ss));
	out.write(reinterpret_cast<char*>(&ww[0]), wSize * sizeof(double));
	out.close();

*/
#endif

#else
	/*
	 * f = x^2 + y^2 + z^2
	 */
	random_device rnde;
	mt19937 mt(rnde());
	uniform_real_distribution<> rnd(-10.0, 10.0);

	Vector3d coordinate = Vector3d::Constant(rnd(mt));
	Vector3d error = Vector3d::Constant(1e-15);
	auto grad = [](Eigen::Vector3d arg) {
		Eigen::Vector3d tmp;
		for(int i=0; i<3; ++i) {
			tmp[i] = 2*arg[i];
		}
		return tmp;
	};
	auto showVec = [](Eigen::Vector3d arg) {
		for(int i=0; i<3; ++i) {
			cout << arg(i) << " ";
		}
		cout << endl;
	};
	auto check = [](Eigen::Vector3d& arg) {
		bool tmp = true;
		for(int i=0; i<3; ++i) {
			if( (std::fabs(arg[i]) < 1e-15) )
				arg[i] = 0.0;
			else
				tmp = false;
		}
		return tmp;
	};

	LearningMethod lmTest(3, 0.1);

	cout << "start" << endl;
	showVec(coordinate);
	cout << endl;

	int count = 0;
	while(1) {
		lmTest.SMD( coordinate, grad(coordinate) );
		++count;
		if(check(coordinate)) {
			cout << "solved : " << count << endl;
			showVec(coordinate);
			break;
		} else if(count > 100000) {
			cout << "count over" << endl;
			break;
		}
		showVec(coordinate);
	}

#endif

	cout << "Learning End" << endl;
	return 0;

}
