// SPDX-License-Identifier: GPL-2.0 OR MIT
/*
 * amdgpu_npu.c — AMD NPU IP Block (XDNA 2)
 *
 * Wraps the amdxdna NPU driver into amdgpu's IP block lifecycle,
 * making the NPU visible as a first-class IP block alongside
 * SDMA, VCN, JPEG, etc.
 *
 * Lifecycle: early_init → sw_init → hw_init → ... → hw_fini → sw_fini
 *
 * This file lives at: drivers/gpu/drm/amd/amdgpu/amdgpu_npu.c
 * Include from:      drivers/gpu/drm/amd/amdgpu/Makefile
 */

#include <linux/firmware.h>
#include <linux/pci.h>
#include <linux/iommu.h>

#include "amdgpu.h"
#include "amdgpu_npu.h"
#include "soc15_common.h"

/* ------------------------------------------------------------------ */
/*  NPU device state (per amdgpu_device)                              */
/* ------------------------------------------------------------------ */

struct amdgpu_npu {
	/* PCI function — the NPU is a separate PCI function on the same
	 * multi-function device as the GPU.  We acquire it via the
	 * auxiliary bus or a bus notifier; for now, stub this out.
	 */
	struct pci_dev		*pdev;

	/* FW loading */
	const struct firmware	*fw;
	char			fw_path[64];

	/* MMIO resources (mapped from NPU PCI BARs) */
	void __iomem		*sram_base;	/* SRAM BAR */
	void __iomem		*mbox_base;	/* Mailbox BAR */
	void __iomem		*psp_regs[4];
	void __iomem		*smu_regs[5];

	/* Firmware interface */
	struct xdna_mailbox	*mbox;
	struct xdna_mailbox_channel *mgmt_chann;

	/* Device status */
	enum { NPU_UNINIT, NPU_STARTED, NPU_STOPPED } status;

	/* Ring for NPU command submission
	 * (mirrors amdgpu_ring.h pattern) */
	struct amdgpu_ring	ring;

	/* IOMMU SVA */
	struct iommu_domain	*domain;
	int			pasid;
};

/* ------------------------------------------------------------------ */
/*  PCI function discovery                                            */
/* ------------------------------------------------------------------ */

/*
 * The NPU on Strix Halo lives at PCI function 1 on the same device
 * as the GPU (function 0).  We discover it by walking the PCI bus
 * from the GPU's pdev.
 *
 * In production this would use the auxiliary bus; for the prototype
 * we do a direct lookup.
 */
static struct pci_dev *amdgpu_npu_find_pci_dev(struct amdgpu_device *adev)
{
	struct pci_dev *gpu_pdev = adev->pdev;

	/* Strix Halo: GPU at <bus>:00.0, NPU at <bus>:00.1 */
	return pci_get_slot(gpu_pdev->bus,
			    PCI_DEVFN(PCI_SLOT(gpu_pdev->devfn), 1));
}

/* ------------------------------------------------------------------ */
/*  Lifecycle: early_init                                             */
/* ------------------------------------------------------------------ */

static int amdgpu_npu_early_init(struct amdgpu_ip_block *ip_block)
{
	struct amdgpu_device *adev = ip_block->adev;
	struct amdgpu_npu *npu;

	npu = kzalloc(sizeof(*npu), GFP_KERNEL);
	if (!npu)
		return -ENOMEM;
	adev->npu = npu;

	npu->status = NPU_UNINIT;

	/* Locate the NPU PCI function */
	npu->pdev = amdgpu_npu_find_pci_dev(adev);
	if (!npu->pdev) {
		dev_info(adev->dev, "NPU PCI function not found\n");
		/* Non-fatal — system may not have NPU */
		return 0;
	}

	dev_info(adev->dev, "NPU found at %s\n", pci_name(npu->pdev));
	return 0;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle: sw_init                                                */
/* ------------------------------------------------------------------ */

static int amdgpu_npu_sw_init(struct amdgpu_ip_block *ip_block)
{
	struct amdgpu_device *adev = ip_block->adev;
	struct amdgpu_npu *npu = adev->npu;
	int ret;

	if (!npu || !npu->pdev)
		return 0; /* No NPU present */

	/* Build firmware path from PCI DID/rev */
	snprintf(npu->fw_path, sizeof(npu->fw_path),
		 "amdnpu/%04x_%02x/npu.dev.sbin",
		 npu->pdev->device, npu->pdev->revision);

	ret = firmware_request_nowarn(&npu->fw, npu->fw_path, &npu->pdev->dev);
	if (ret) {
		dev_err(adev->dev, "NPU firmware %s not found\n", npu->fw_path);
		return ret;
	}

	/* Map NPU PCI BARs — follows same pattern as aie2_init() */
	// ... BAR mapping from amdxdna ...

	/* Init NPU ring (amdgpu_ring pattern) */
	ret = amdgpu_ring_init(adev, &npu->ring, 1024, NULL, 0,
			       AMDGPU_RING_PRIO_DEFAULT, NULL);
	if (ret) {
		release_firmware(npu->fw);
		return ret;
	}

	dev_info(adev->dev, "NPU firmware loaded: %s\n", npu->fw_path);
	return 0;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle: hw_init                                                */
/* ------------------------------------------------------------------ */

static int amdgpu_npu_hw_init(struct amdgpu_ip_block *ip_block)
{
	struct amdgpu_device *adev = ip_block->adev;
	struct amdgpu_npu *npu = adev->npu;

	if (!npu || !npu->pdev)
		return 0;

	/* === This is where amdxdna's aie2_init() logic goes === */
	//
	// 1. Enable PCI + bus master
	//    pci_enable_device(npu->pdev);
	//    pci_set_master(npu->pdev);
	//
	// 2. Map BARs: SRAM, mailbox, PSP, SMU
	//    (same as aie2_init in amdxdna)
	//
	// 3. Init PSP → load NPU firmware
	//    aie_psp_start(npu->psp);
	//
	// 4. Init SMU → power on NPU
	//    aie_smu_init(npu->smu);
	//    (note: PSP first, then SMU — see fix in amdxdna)
	//
	// 5. Wait for firmware alive on SRAM
	//    aie2_get_mgmt_chann_info(npu);
	//
	// 6. Start mailbox channel
	//    xdna_mailbox_start_channel(npu->mgmt_chann, ...);
	//
	// 7. Query firmware version + AIE metadata
	//    aie2_mgmt_fw_query(npu);

	npu->status = NPU_STARTED;
	dev_info(adev->dev, "NPU started\n");
	return 0;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle: hw_fini                                                */
/* ------------------------------------------------------------------ */

static int amdgpu_npu_hw_fini(struct amdgpu_ip_block *ip_block)
{
	struct amdgpu_device *adev = ip_block->adev;
	struct amdgpu_npu *npu = adev->npu;

	if (!npu || !npu->pdev || npu->status != NPU_STARTED)
		return 0;

	/* === Teardown: reverse of hw_init === */
	// 1. Suspend firmware
	// 2. Stop mailbox channel
	// 3. Stop PSP
	// 4. SMU power off
	// 5. Disable PCI

	npu->status = NPU_STOPPED;
	return 0;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle: sw_fini                                                */
/* ------------------------------------------------------------------ */

static int amdgpu_npu_sw_fini(struct amdgpu_ip_block *ip_block)
{
	struct amdgpu_device *adev = ip_block->adev;
	struct amdgpu_npu *npu = adev->npu;

	if (!npu)
		return 0;

	amdgpu_ring_fini(&npu->ring);
	if (npu->fw)
		release_firmware(npu->fw);
	if (npu->pdev)
		pci_dev_put(npu->pdev);
	kfree(npu);
	adev->npu = NULL;
	return 0;
}

/* ------------------------------------------------------------------ */
/*  Power management hooks                                            */
/* ------------------------------------------------------------------ */

static int amdgpu_npu_suspend(struct amdgpu_ip_block *ip_block)
{
	return amdgpu_npu_hw_fini(ip_block);
}

static int amdgpu_npu_resume(struct amdgpu_ip_block *ip_block)
{
	return amdgpu_npu_hw_init(ip_block);
}

/* ------------------------------------------------------------------ */
/*  IP funcs vtable                                                   */
/* ------------------------------------------------------------------ */

static const struct amd_ip_funcs amdgpu_npu_ip_funcs = {
	.name = "amdgpu_npu",
	.early_init = amdgpu_npu_early_init,
	.sw_init    = amdgpu_npu_sw_init,
	.hw_init    = amdgpu_npu_hw_init,
	.hw_fini    = amdgpu_npu_hw_fini,
	.sw_fini    = amdgpu_npu_sw_fini,
	.suspend    = amdgpu_npu_suspend,
	.resume     = amdgpu_npu_resume,
};

/* ------------------------------------------------------------------ */
/*  IP block version descriptor                                       */
/* ------------------------------------------------------------------ */

const struct amdgpu_ip_block_version npu_ip_block = {
	.type  = AMD_IP_BLOCK_TYPE_NPU,
	.major = 1,
	.minor = 0,
	.rev   = 0,
	.funcs = &amdgpu_npu_ip_funcs,
};

/* ================================================================== */
/*  Integration points (stubs — need changes in 3 files)             */
/* ================================================================== */
/*
 * 1. include/linux/amdgpu_ip.h
 *    Add AMD_IP_BLOCK_TYPE_NPU to enum amd_ip_block_type
 *    (before AMD_IP_BLOCK_TYPE_NUM)
 *
 * 2. drivers/gpu/drm/amd/amdgpu/Makefile
 *    amdgpu-y += amdgpu_npu.o
 *
 * 3. amdgpu_device.c
 *    In amdgpu_device_ip_early_init():
 *      amdgpu_device_ip_block_add(adev, &npu_ip_block);
 *
 *    For IP-discovery ASICs: add NPU hw_id to hw_id_map[] in
 *    amdgpu_discovery.c
 *
 * 4. include/drm/amdgpu.h
 *    Add to struct amdgpu_device:
 *      struct amdgpu_npu *npu;
 */
