#pragma once

#include <concepts>

#include "vector.hpp"

namespace terra::linalg {

/// @brief Concept for types that behave like linear forms.
///
/// Evaluates a linear form into a vector.
///
/// This could be something like
///
/// \f[ L(v_h) = \int f v_h \f]
///
/// evaluated into the entries \f$q_k\f$ of a coefficient vector \f$q\f$:
///
/// \f[ q_k = \int f \phi_k \f].
///
template < typename T >
concept LinearFormLike = requires( T& self, typename T::DstVectorType& dst ) {
    // Requires exposing the vector types.
    typename T::DstVectorType;

    // Require that dst vector type satisfies VectorLike
    requires VectorLike< typename T::DstVectorType >;

    // Required evaluation implementation
    { self.apply_impl( dst ) } -> std::same_as< void >;
};

/// @brief Apply a linear form and write to a destination vector.
/// @param L Linear form to apply.
/// @param dst Destination vector.
template < LinearFormLike LinearForm >
void apply( LinearForm& L, typename LinearForm::DstVectorType& dst )
{
    L.apply_impl( dst );
}

} // namespace terra::linalg