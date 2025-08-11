/*
 * @Author       : LIN Guocheng
 * @Date         : 2024-05-14 17:42:23
 * @LastEditors  : LIN Guocheng
 * @LastEditTime : 2024-10-09 01:17:13
 * @FilePath     : /root/PPFP/modules/ipv4/RlPpfpRoutingTable.h
 * @Description  : Probabilistic routing table in RouterRL.
 */

#ifndef RLPPFPROUTINGTABLE_H
#define RLPPFPROUTINGTABLE_H
#include "RlProbabilisticRoutingTable.h"

/**
 * Stores the forwarding probabilities for the entire network and serves as the network's statistics module, exchanging data with the Python side through ZMQ communication.
 * Currently, there is no method for using it as a local variable, and there is only a single global static object.
 * In a multi-agent environment, it can be extended to one object per node.
 */
class RlPpfpRoutingTable : public RlProbabilisticRoutingTable
{
public:
    RlPpfpRoutingTable();
    ~RlPpfpRoutingTable();
    /**
     * Used to get the unique static instance, throws an exception if the instance is not initialized.
     */
    static RlPpfpRoutingTable *getInstance();

    /**
     * Used to initialize the unique static instance, set parameters, and allocate memory for statistics variables.
     */
    static RlPpfpRoutingTable *initTable(int num, const char *file, int port, double overTime_v,
                                         int totalStep_v, int simMode_v);

    /**
     * Initializes the instance, establishes ZMQ communication with the Python side based on TCP, and allocates memory for statistics variables.
     */
    void initiate() override;

    void updateRoutingTable(int step, double stepTime) override;
    int getNextNode(int nodeId, int srcNode, int dstNode) override;

protected:
    int **topoDstTable;  // Distance table in topology

private:
    static RlPpfpRoutingTable *ppfpRoutingTable;
};

#endif // RLPPFPROUTINGTABLE_H