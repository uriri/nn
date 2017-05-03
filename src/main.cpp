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
#include <bitset>
#include <vector>
#include <memory>
#include <array>
#include <random>
#include <cmath>

#include "NeuralNetwork.h"
#include "arrayFunc.h"
#include "MyClocks.h"

//#define DEBUG
//
//相対強さ用の学習
#define RELATIVE

//plotファイル出力
#define LOGPLOT

//SMDを使う（たぶんしない）
//#define USE_SMD
#ifdef USE_SMD
#include "gradient/SMD.h"
#else
#include "gradient/Adam.h"
#endif

#define printVar(var) std::cout << #var" : " << var << std::endl

#ifdef RELATIVE
constexpr std::size_t inSize = 14;
#else
constexpr std::size_t inSize = 64;
#endif
constexpr std::size_t hideSize = 20;
constexpr std::size_t outSize = 1;
constexpr std::size_t wSize = ((inSize + 1) * hideSize) + (hideSize + 1) * outSize;

//学習用データを4:1で学習、テストに分割
constexpr double use4Learn = 4.0;
constexpr double use4Test = 1.0;
constexpr double division = use4Learn+use4Test;

//loss function
constexpr double  alpha = 1.0;//シグモイド関数のゲイン

//損失関数、中身はシグモイド
constexpr double lossFunc(const double arg){
	return ( 1.0/(1.0+std::exp(-alpha*arg)) );
}
constexpr double d_LossFunc(const double arg){
	return ( alpha*lossFunc(arg)*(1.0-lossFunc(arg)) );
}

template<class T>
void printArray(const T& arg) {
	for(const auto& a : arg)
		std::cout << a << " ";
	std::cout << std::endl;
}

std::string makeFileName(const std::string& s) {
	unsigned int num = 0;
	while (1) {
		std::ostringstream os;
		os << "plot_" << s << num << ".csv";
		std::ifstream check(os.str().c_str());
		if (check.fail()) {
			return os.str();
		}
		++num;
	}
}

//ミニバッチ用訓練データ
template<std::size_t size>
struct Data {
	std::array<int, size> bestHandRank;	//手札配列
	std::vector<std::array<int, size>> otherHandRank;	//[数][手札配列]
	void setOtherHandQty(int qty) {
		otherHandRank.reserve(qty);
	}
};

using Data_ptr = std::unique_ptr<Data<inSize>>;

//ファイルから読み込んだログデータをvectorに格納
void readLogFile(const std::string& fileName, std::vector<Data_ptr>& readData){
	std::ifstream ifs;
	ifs.open(fileName.c_str(), std::ios::binary);

	if(ifs.fail()){
		return;
	}

	//ファイルの終わりまでループ
	while (!ifs.eof()) {
		Data_ptr nowData = std::make_unique<Data<inSize>>();
		int otherMvQty;
		ifs.read(reinterpret_cast<char*>(&otherMvQty), sizeof(otherMvQty));
		nowData->setOtherHandQty(otherMvQty);

#ifdef RELATIVE
		int bestHandRank[inSize];
		ifs.read(reinterpret_cast<char*>(bestHandRank), sizeof(bestHandRank));
#else
		int bestHandRank[inSize], otherHandRank[inSize];
		unsigned long long int bestHand, otherHand;
		ifs.read(reinterpret_cast<char*>(&bestHand), sizeof(bestHand));
#endif
		nowData->bestHandRank = toArray(bestHandRank);

		for (int j=0; j<otherMvQty; ++j) {
#ifdef RELATIVE
			int otherHandRank[inSize];
			ifs.read(reinterpret_cast<char*>(otherHandRank), sizeof(otherHandRank));
#else
			int bestHandRank[inSize], otherHandRank[inSize];
			unsigned long long int bestHand, otherHand;
			ifs.read(reinterpret_cast<char*>(&otherHand), sizeof(otherHand));
#endif
			nowData->otherHandRank.emplace_back(toArray(otherHandRank));
		}//other hand end
		readData.push_back(std::move(nowData));
	} //Learn File end

	ifs.close();
};

//bonanza method
//学習からテストを行う
//一つの関数でいろいろやってて見にくい -> 処理を見直した方がいいかも
//2017/03/28 ファイル読み込みだけ分けた
//2017/05/03 一つのログファイルを学習用とテスト用に分割する使用に変更
void learnWithBonanza(const std::string& fileName, bool isNF, unsigned int learningTimes) {
	using namespace std;
	using namespace Eigen;

	NN::NeuralNetwork<inSize, hideSize, outSize> nn;

//	printVar(wSize);

#ifdef USE_SMD
	//SMD parameter
	constexpr double lamda = 0.0;
	constexpr double learningRate = 0.09;
	constexpr double mu = 0.05;			//meta-LearningRate
	unique_ptr<Learn::GradientDescent> method = std::make_unique<Learn::SMD>(wSize, learningRate, mu, lamda);
#else
	unique_ptr<Learn::GradientDescent> method = std::make_unique<Learn::Adam>(wSize);
#endif

	auto shuffleDataSet = [](std::vector<Data_ptr>& ds){
		std::random_device rnd;
		std::mt19937 mt(rnd());
		std::shuffle(ds.begin(), ds.end(), mt);
	};

	vector<Data_ptr> dataSet4Learn;
	vector<Data_ptr> dataSet4Test;

	{
		//reading dataset
		vector<Data_ptr> dataSet;
		readLogFile(fileName, dataSet);
		if(dataSet.empty()){
			printVar(fileName);
			std::cerr << "data set is empty" << std::endl;
			return;
		}

		shuffleDataSet(dataSet);

		//divide the data set for learning and testing
		const auto size4Learn = (dataSet.size()/division)*use4Learn;

		dataSet4Learn.insert(
				std::end(dataSet4Learn),
				std::make_move_iterator(std::begin(dataSet)),
				std::make_move_iterator(std::begin(dataSet)+size4Learn)
		);

		dataSet4Test.insert(
				std::end(dataSet4Test),
				std::make_move_iterator(std::begin(dataSet)+size4Learn),
				std::make_move_iterator(std::end(dataSet))
		);
	}

#ifdef LOGPLOT
	ofstream plot;
	string plotFile = makeFileName((isNF?"NF":"RF"));
	cout << plotFile << endl;
	plot.open(plotFile.c_str());
#endif

	//for learning
	unsigned int notChangeCount = 0;
	double beforeLoss = numeric_limits<double>::max();
	double minLossValue = numeric_limits<double>::max();

	//for plot
	constexpr unsigned int geneNum = 10;	//群平均の分母
	double aveMatch = 0.0;					//平均

	//最良の重み
	MatrixXd bestI2H, bestH2O;

	for (unsigned int itr=0; itr<learningTimes; ++itr) {

		double totalLoss = 0.0;
		VectorXd lossGrad = VectorXd::Zero(wSize);	//損失勾配

		//イテレーションごとにデータセットをシャッフル
		shuffleDataSet(dataSet4Learn);

		//学習用データセット
		for(const auto& data : dataSet4Learn){
			const auto& best = data->bestHandRank;
			nn.setInput(best);
			const double bestValue = nn.getOutPut();
			VectorXd bestGrad;
			nn.getGrad(bestGrad);

#ifdef DEBUG
			cout << "bestHand" << endl;
			printArray(best);
			cout << "bestValue  " << bestValue << endl;
#endif

			const auto& others = data->otherHandRank;
			for (const auto& other : others) {
				nn.setInput(other);

				const double otherValue = nn.getOutPut();
				VectorXd otherGrad;
				nn.getGrad(otherGrad);

				//勾配をここで求めて総和を計算
				const double lossValue = otherValue - bestValue;
				totalLoss += lossFunc(lossValue);
				lossGrad += ( d_LossFunc(lossValue) * (otherGrad - bestGrad) );

#ifdef DEBUG
				cout << "otherHand" << endl;
				printArray(other);
				cout << "otherValue " << otherValue << endl;
				cout << "loss       " << lossFunc(lossValue) << "(" << lossValue << ") "
						<< d_LossFunc(lossValue) << endl;
#endif

			}//other hand end
		}//end learn data set

		cout << itr << " : " << beforeLoss << "->" << totalLoss << " (" << totalLoss - beforeLoss << ")" << endl;

		if (std::fabs(totalLoss - beforeLoss) > 1e-15) {
			beforeLoss = totalLoss;
			notChangeCount = 0;
		} else {
			++notChangeCount;
			cout << "not change " << notChangeCount << endl;
		}

		//終了条件
		if (notChangeCount >= 10) {
			cout << "stop learn" << endl;
			break;
		}
		if (std::fabs(totalLoss) < 1e-15) {
			cout << "Optimized totalLoss" << endl;
			minLossValue = totalLoss;
			bestI2H = nn.getWeightI2H_Mat();
			bestH2O = nn.getWeightI2H_Mat();
			break;
		}

		//最良値の更新
		if (totalLoss < minLossValue) {
			minLossValue = totalLoss;
			bestI2H = nn.getWeightI2H_Mat();
			bestH2O = nn.getWeightI2H_Mat();
		}

		//勾配法による学習
		VectorXd wVec;
		nn.getWeightOneDim(wVec);
		method->execute(wVec, lossGrad);
		nn.setWeight(wVec);

		/*******
		 * test
		 *******/

		unsigned int bestCount = 0;
		const unsigned int count = dataSet4Test.size();

		//テスト用データセット
		for(const auto& data : dataSet4Test){
			bool isBest = true;

			const auto& best = data->bestHandRank;
			nn.setInput(best);
			const double bestValue = nn.getOutPut();

			const auto& others = data->otherHandRank;
			for (const auto& other : others) {
				nn.setInput(other);
				const double otherValue = nn.getOutPut();
				if (otherValue < bestValue) {
					//事前に読み込むようにしたのでbreakしても大丈夫
					isBest = false;
					break;
				}
			}

			if (isBest) {
				++bestCount;
			}
		}//test File end

		//汎化テストの一致率
		const double match = (static_cast<double>(bestCount)/count) * 100;
		cout << "     " << bestCount << "/" << count << " " << match << "%" << endl;

#ifdef  LOGPLOT
		plot << itr << "," << totalLoss << "," << match;

		//群平均
		aveMatch += match;

		if(itr%geneNum != 0) {
			plot << ",=NA()";
		} else {
			plot << "," << aveMatch/geneNum;
			aveMatch = 0.0;
		}
#ifdef USE_SMD
		plot << "," << method.getLearningRateNorm();
		if(itr == 1) {
			plot << ",," << "rate|meta(" << learningRate << "|" << mu << ")," << nn.getSeed();
		}
#endif
		plot << endl;
#endif

	} //learning loop

	//損失が一番小さい重みを書き出す
	ostringstream out;
#ifdef RELATIVE
	out << "RelativeStrong/weight/" << (isNF ? "NF_" : "RF_") << hideSize;
#else
	out << "Normal/weight/" << (isNF?"NF_":"RF_") << hideSize << "_" << minLossValue;
#endif
	Eigen::writeBinary(out.str()+"_I2H_" + std::to_string(minLossValue) + ".dat", bestI2H);
	Eigen::writeBinary(out.str()+"_H2O_" + std::to_string(minLossValue) + ".dat", bestH2O);
}

//反復回数を変えるたびにコンパイルするのめんどくさいのでファイル読み込み
void readLearnParameter(unsigned int& learningTimes, std::string& fileRF, std::string& fileNF){
	std::ifstream ifs("parameter.dat");
	std::string dir;
	while(!ifs.eof()){
		std::string param, val;
		ifs >> param >> val;
		if(param=="Dir"){
			dir = val;
		}else if(param=="FileRF"){
			fileRF = val;
		}else if(param=="FileNF"){
			fileNF = val;
		}else if(param=="iteration"){
			learningTimes = std::stoi(val);
		} else {
			std::cout << "unkown " << param << std::endl;
		}
	}
	fileRF  = dir + fileRF;
	fileNF  = dir + fileNF;
}

int main() {
	unsigned int learningTimes;
	std::string fileRF, fileNF;

	readLearnParameter(learningTimes, fileRF, fileNF);

	learnWithBonanza(fileRF, false, learningTimes);
	learnWithBonanza(fileNF, true, learningTimes);

	std::cout << "Learning End" << std::endl;
	return 0;
}
