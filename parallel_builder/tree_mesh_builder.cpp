/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::buildOctree(
    const float edgeSize,
    const Vec3_t<float> &startBlock,
    const ParametricScalarField &field
)
{
    float edgeHalf = edgeSize / 2.0f;

    // The middle of the current block transformed to be usable in evaluateFieldAt()
    Vec3_t<float> blockMiddle(
        (startBlock.x + edgeHalf) * mGridResolution,
        (startBlock.y + edgeHalf) * mGridResolution,
        (startBlock.z + edgeHalf) * mGridResolution
    );

    float eval = evaluateFieldAt(blockMiddle, field);

    bool blockEmpty = eval > mIsoLevel + (emptinessInequalityConstant * (edgeSize * mGridResolution));
    if (blockEmpty) {
        return 0;
    }

    // If smaller than cutoff, we are done and can buildCubes()
    if (edgeHalf < 1) {
        return buildCube(startBlock, field);
    } else {
        // The 8 children
        unsigned trianglesCount = 0;
        trianglesCount += buildOctree(edgeHalf, Vec3_t<float> (startBlock.x, startBlock.y, startBlock.z), field);
        trianglesCount += buildOctree(edgeHalf, Vec3_t<float> (startBlock.x + edgeHalf, startBlock.y, startBlock.z), field);
        trianglesCount += buildOctree(edgeHalf, Vec3_t<float> (startBlock.x, startBlock.y + edgeHalf, startBlock.z), field);
        trianglesCount += buildOctree(edgeHalf, Vec3_t<float> (startBlock.x, startBlock.y, startBlock.z + edgeHalf), field);
        trianglesCount += buildOctree(edgeHalf, Vec3_t<float> (startBlock.x + edgeHalf, startBlock.y + edgeHalf, startBlock.z), field);
        trianglesCount += buildOctree(edgeHalf, Vec3_t<float> (startBlock.x + edgeHalf, startBlock.y, startBlock.z + edgeHalf), field);
        trianglesCount += buildOctree(edgeHalf, Vec3_t<float> (startBlock.x, startBlock.y + edgeHalf, startBlock.z + edgeHalf), field);
        trianglesCount += buildOctree(edgeHalf, Vec3_t<float> (startBlock.x + edgeHalf, startBlock.y + edgeHalf, startBlock.z + edgeHalf), field);

        return trianglesCount;
    }

    // Shouldn't occur
    return 0;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    float edgeSize = field.getSize().x; // Is a cube - one edge is enough
    Vec3_t<float> startBlock(0); // The starting position of the initial undivided block

    unsigned trianglesCount = buildOctree(mGridSize, startBlock, field);

    return trianglesCount;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    mTriangles.push_back(triangle);
}
