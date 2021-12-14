/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Patrik Nemeth <xnemet04@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    14.12.2021
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
    const Vec3_t<float> &currentBlock,
    const ParametricScalarField &field
)
{
    const float edgeHalf = edgeSize / 2.0f;

    if (edgeHalf < 1) {
        // We are done and can buildCubes()
        return buildCube(currentBlock, field);
    }

    // The middle of the current block transformed to be usable in evaluateFieldAt()
    Vec3_t<float> blockMiddle(
        (currentBlock.x + edgeHalf) * mGridResolution,
        (currentBlock.y + edgeHalf) * mGridResolution,
        (currentBlock.z + edgeHalf) * mGridResolution
    );

    float eval = evaluateFieldAt(blockMiddle, field);

    bool blockEmpty = eval > mIsoLevel + (emptinessInequalityConstant * (edgeSize * mGridResolution));
    if (blockEmpty) {
        // If the current block does not generate a surface, return
        return 0;
    }
    // Otherwise generate next 8 children
    unsigned trianglesCount = 0;

    for (unsigned i = 0; i < 8; i++) {
        #pragma omp task shared(trianglesCount)
        {
            unsigned tmp = buildOctree(
                edgeHalf,
                Vec3_t<float> (
                    currentBlock.x + (edgeHalf * ((i & 0b001) > 0)),
                    currentBlock.y + (edgeHalf * ((i & 0b010) > 0)),
                    currentBlock.z + (edgeHalf * ((i & 0b100) > 0))
                ),
                field);

            #pragma omp critical
            trianglesCount += tmp;
        }
    }

    #pragma omp taskwait
    return trianglesCount;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    Vec3_t<float> startBlock(0); // The starting position of the initial undivided block

    unsigned trianglesCount;

    #pragma omp parallel
    #pragma omp single
    trianglesCount = buildOctree(mGridSize, startBlock, field);

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
    #pragma omp critical
    mTriangles.push_back(triangle);
}
