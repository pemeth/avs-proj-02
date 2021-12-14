/**
 * @file    tree_mesh_builder.h
 *
 * @author  Patrik Nemeth <xnemet04@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    14.12.2021
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include <math.h>

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    unsigned buildOctree(const float edgeSize, const Vec3_t<float> &currentBlock, const ParametricScalarField &field);

    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }

    std::vector<Triangle_t> mTriangles; ///< Temporary array of triangles
    const float emptinessInequalityConstant = sqrt(3) / 2.0f; ///< See equation 5.3 in the assignment
};

#endif // TREE_MESH_BUILDER_H
