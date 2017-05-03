/*
 * ClockMS.h
 *
 *  Created on: 2016/05/30
 *      Author: haga
 */

#ifndef MYCLOCKS_H_
#define MYCLOCKS_H_

#include <chrono>

class MyClock {
	using scs_c = std::chrono::system_clock;
private:
	scs_c::time_point s;
public:
	using second = std::chrono::seconds;
	using millis = std::chrono::milliseconds;
	using micros = std::chrono::microseconds;
	void start() {
		s = scs_c::now();
	}
	template<typename T>
	long stop() {
		auto e = scs_c::now();
		long t = std::chrono::duration_cast<T>(e - s).count();
		return t;
	}
	template<typename T>
	long restart() {
		auto e = scs_c::now();
		long t = std::chrono::duration_cast<T>(e - s).count();
		s = e;
		return t;
	}
	MyClock(){
		start();
	};
	virtual ~MyClock(){};
};

#if 0
class ClockMicS {
	//microsec単位
private:
	timeval t_start;
public:
	void start() {gettimeofday(&t_start,NULL);}
	long stop() {
		timeval t_end;
		gettimeofday(&t_end,NULL);
		long t=(t_end.tv_sec-t_start.tv_sec)*1000000+t_end.tv_usec-t_start.tv_usec;
		return t;
	}
	long restart() { //結果を返し、0から再スタート
		timeval t_end;
		gettimeofday(&t_end,NULL);
		long t=(t_end.tv_sec-t_start.tv_sec)*1000000+t_end.tv_usec-t_start.tv_usec;
		t_start=t_end;
		return t;
	}
	ClockMicS() {}
	ClockMicS(int m) {start();}
};
#endif

#endif /* MYCLOCKS_H_ */
