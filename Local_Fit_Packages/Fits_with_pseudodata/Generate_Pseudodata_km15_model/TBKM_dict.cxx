// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME TBKM_dict
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "ROOT/RConfig.hxx"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Header files passed as explicit arguments
#include "TBKM.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace ROOT {
   static void *new_TBKM(void *p = nullptr);
   static void *newArray_TBKM(Long_t size, void *p);
   static void delete_TBKM(void *p);
   static void deleteArray_TBKM(void *p);
   static void destruct_TBKM(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::TBKM*)
   {
      ::TBKM *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::TBKM >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("TBKM", ::TBKM::Class_Version(), "TBKM.h", 10,
                  typeid(::TBKM), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::TBKM::Dictionary, isa_proxy, 4,
                  sizeof(::TBKM) );
      instance.SetNew(&new_TBKM);
      instance.SetNewArray(&newArray_TBKM);
      instance.SetDelete(&delete_TBKM);
      instance.SetDeleteArray(&deleteArray_TBKM);
      instance.SetDestructor(&destruct_TBKM);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::TBKM*)
   {
      return GenerateInitInstanceLocal(static_cast<::TBKM*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::TBKM*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
atomic_TClass_ptr TBKM::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *TBKM::Class_Name()
{
   return "TBKM";
}

//______________________________________________________________________________
const char *TBKM::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::TBKM*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int TBKM::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::TBKM*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *TBKM::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::TBKM*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *TBKM::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::TBKM*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
void TBKM::Streamer(TBuffer &R__b)
{
   // Stream an object of class TBKM.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(TBKM::Class(),this);
   } else {
      R__b.WriteClassBuffer(TBKM::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_TBKM(void *p) {
      return  p ? new(p) ::TBKM : new ::TBKM;
   }
   static void *newArray_TBKM(Long_t nElements, void *p) {
      return p ? new(p) ::TBKM[nElements] : new ::TBKM[nElements];
   }
   // Wrapper around operator delete
   static void delete_TBKM(void *p) {
      delete (static_cast<::TBKM*>(p));
   }
   static void deleteArray_TBKM(void *p) {
      delete [] (static_cast<::TBKM*>(p));
   }
   static void destruct_TBKM(void *p) {
      typedef ::TBKM current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::TBKM

namespace {
  void TriggerDictionaryInitialization_TBKM_dict_Impl() {
    static const char* headers[] = {
"TBKM.h",
nullptr
    };
    static const char* includePaths[] = {
"/sfs/gpfs/tardis/applications/202506/software/standard/mpi/gcc/11.4.0/openmpi/4.1.4/root/6.32.06/include/",
"/sfs/gpfs/tardis/home/qzf7nj/Physics_Testing/km15_implementation/updated_km15_implementation/",
nullptr
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "TBKM_dict dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
class __attribute__((annotate("$clingAutoload$TBKM.h")))  TBKM;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "TBKM_dict dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "TBKM.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"TBKM", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("TBKM_dict",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_TBKM_dict_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_TBKM_dict_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_TBKM_dict() {
  TriggerDictionaryInitialization_TBKM_dict_Impl();
}
