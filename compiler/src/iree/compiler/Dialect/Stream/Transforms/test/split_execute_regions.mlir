// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-stream-split-execute-regions))" %s | FileCheck %s

// Tests splitting a merged execute region with two sequential dispatches.
// Ensures that data dependencies between dispatches are correctly routed through
// the `with(...)` captures of the newly split regions.

// CHECK-LABEL: @splitTwoDispatches
util.func public @splitTwoDispatches(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c1280 = arith.constant 1280 : index
  %c255_i32 = arith.constant 255 : i32

  // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.async.execute
  // CHECK-SAME: with(%arg0 as %[[A_CAP0:.+]]: !stream.resource<external>{%c20})
  // CHECK-SAME: -> !stream.resource<transient>{%c1280} {
  // CHECK-NEXT:   %[[SPLAT:.+]] = stream.async.splat %c255_i32
  // CHECK-NEXT:   %[[D0:.+]] = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%[[A_CAP0]][{{.+}}], %[[SPLAT]][{{.+}}])
  // CHECK-NEXT:   stream.yield %[[D0]] : !stream.resource<transient>{%c1280}
  // CHECK-NEXT: } => !stream.timepoint

  // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.async.execute
  // CHECK-SAME: await(%[[TP0]]) => with(
  // CHECK-SAME: %[[R0]] as %[[D0_CAP:.+]]: !stream.resource<transient>{%c1280},
  // CHECK-SAME: %arg0 as %[[A_CAP1:.+]]: !stream.resource<external>{%c20})
  // CHECK-SAME: -> !stream.resource<external>{%c20} {
  // CHECK-NEXT:   %[[D1:.+]] = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%[[D0_CAP]][{{.+}}], %[[A_CAP1]][{{.+}}])
  // CHECK-NEXT:   stream.yield %[[D1]] : !stream.resource<external>{%c20}
  // CHECK-NEXT: } => !stream.timepoint

  %results:2, %tp = stream.async.execute
      with(%arg0 as %a: !stream.resource<external>{%c20})
      -> (!stream.resource<transient>{%c1280}, !stream.resource<external>{%c20}) {
    %splat = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
    %d0 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%a[%c0 to %c20 for %c20], %splat[%c0 to %c1280 for %c1280]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c1280}) -> %splat{%c1280}
    %d1 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%d0[%c0 to %c1280 for %c1280], %a[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c1280}, !stream.resource<external>{%c20}) -> !stream.resource<external>{%c20}
    stream.yield %d0, %d1 : !stream.resource<transient>{%c1280}, !stream.resource<external>{%c20}
  } => !stream.timepoint

  // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TP1]] => %[[R1]]
  %ready = stream.timepoint.await %tp => %results#1 : !stream.resource<external>{%c20}
  // CHECK-NEXT: util.return %[[READY]]
  util.return %ready : !stream.resource<external>
}

// -----

// Tests that single-dispatch execute regions are left untouched.

// CHECK-LABEL: @singleDispatchUntouched
util.func public @singleDispatchUntouched(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_0
  // CHECK-NEXT: stream.yield
  // CHECK-NEXT: } => !stream.timepoint
  // CHECK-NOT: stream.async.execute
  %result, %tp = stream.async.execute
      with(%arg0 as %a: !stream.resource<external>{%c20})
      -> !stream.resource<external>{%c20} {
    %d0 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%a[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}) -> !stream.resource<external>{%c20}
    stream.yield %d0 : !stream.resource<external>{%c20}
  } => !stream.timepoint
  %ready = stream.timepoint.await %tp => %result : !stream.resource<external>{%c20}
  util.return %ready : !stream.resource<external>
}

// -----

// Tests that support ops (like splats) are correctly pushed into the specific
// split execute region that actually needs them, rather than being left behind.

// CHECK-LABEL: @supportOpPlacement
util.func public @supportOpPlacement(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c255_i32 = arith.constant 255 : i32

  // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.async.execute
  // CHECK-NOT: stream.async.splat
  // CHECK: stream.async.dispatch @ex::@dispatch_0
  
  // CHECK: stream.async.execute
  // CHECK-SAME: await(%[[TP0]]) =>
  // CHECK: %[[SPLAT:.+]] = stream.async.splat
  // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1]({{.+}}, %[[SPLAT]][{{.+}}])

  %results:2, %tp = stream.async.execute
      with(%arg0 as %a: !stream.resource<external>{%c20})
      -> (!stream.resource<transient>{%c20}, !stream.resource<external>{%c20}) {
    
    // This splat is ONLY used by dispatch_1.
    %splat = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c20}
    
    %d0 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%a[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}) -> !stream.resource<transient>{%c20}
    %d1 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%d0[%c0 to %c20 for %c20], %splat[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c20}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}
    stream.yield %d0, %d1 : !stream.resource<transient>{%c20}, !stream.resource<external>{%c20}
  } => !stream.timepoint

  %ready = stream.timepoint.await %tp => %results#1 : !stream.resource<external>{%c20}
  util.return %ready : !stream.resource<external>
}

// -----

// Tests that if the original execute region already awaited a timepoint,
// the first split region inherits that await properly.

// CHECK-LABEL: @initialAwaitPropagation
util.func public @initialAwaitPropagation(%arg0: !stream.resource<external>, %initial_tp: !stream.timepoint) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index

  // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.async.execute
  // CHECK-SAME: await(%initial_tp) =>
  // CHECK: stream.async.dispatch @ex::@dispatch_0
  
  // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.async.execute
  // CHECK-SAME: await(%[[TP0]]) =>
  // CHECK: stream.async.dispatch @ex::@dispatch_1

  // Note the '=>' added here to fix the parsing error!
  %results:2, %tp = stream.async.execute await(%initial_tp) =>
      with(%arg0 as %a: !stream.resource<external>{%c20})
      -> (!stream.resource<transient>{%c20}, !stream.resource<external>{%c20}) {
    %d0 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%a[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}) -> !stream.resource<transient>{%c20}
    %d1 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%d0[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}
    stream.yield %d0, %d1 : !stream.resource<transient>{%c20}, !stream.resource<external>{%c20}
  } => !stream.timepoint

  %ready = stream.timepoint.await %tp => %results#1 : !stream.resource<external>{%c20}
  util.return %ready : !stream.resource<external>
}