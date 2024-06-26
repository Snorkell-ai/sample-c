#include <string.h>
#include <stdio.h>
#include <zephyr/kernel.h>
#include <modem/sms.h>
#include <zephyr/logging/log.h>

#include "sms_deliver.h"
#include "sms_internal.h"
#include "parser.h"
#include "string_conversion.h"

/**
 * @brief Decode first byte of SMS-DELIVER header as specified in 3GPP TS 23.040 Section 9.2.2.1.
 *
 * @param[in,out] parser Parser instance.
 * @param[in] buf Buffer containing PDU and pointing to this field.
 *
 * @return Number of parsed bytes.
 */
static int decode_pdu_deliver_header(struct parser *parser, uint8_t *buf)
{
	struct pdu_deliver_data * const pdata = parser->data;

	pdata->header = *((struct pdu_deliver_header *)buf);

	LOG_DBG("SMS header 1st byte: 0x%02X", *buf);

	LOG_DBG("TP-Message-Type-Indicator: %d", pdata->header.mti);
	LOG_DBG("TP-More-Messages-to-Send: %d", pdata->header.mms);
	LOG_DBG("TP-Status-Report-Indication: %d", pdata->header.sri);
	LOG_DBG("TP-User-Data-Header-Indicator: %d", pdata->header.udhi);
	LOG_DBG("TP-Reply-Path: %d", pdata->header.rp);

	return 1;
}

/**
 * @brief Decode TP-Originating-Address as specified in 3GPP TS 23.040 Section 9.2.3.7 and 9.1.2.5.
 *
 * @param[in,out] parser Parser instance.
 * @param[in] buf Buffer containing PDU and pointing to this field.
 *
 * @return Number of parsed bytes.
 */
static int decode_pdu_oa_field(struct parser *parser, uint8_t *buf)
{
	struct pdu_deliver_data * const pdata = parser->data;
	uint8_t address[SMS_MAX_ADDRESS_LEN_OCTETS];
	uint8_t length;

	pdata->oa.length = *buf++;
	pdata->oa.type = (uint8_t)*buf++;

	LOG_DBG("Address-Length: %d", pdata->oa.length);
	LOG_DBG("Type-of-Address: 0x%02X", pdata->oa.type);

	if (pdata->oa.length > SMS_MAX_ADDRESS_LEN_CHARS) {
		LOG_ERR("Maximum address length (%d) exceeded %d. Aborting decoding.",
			SMS_MAX_ADDRESS_LEN_CHARS,
			pdata->oa.length);
		return -EINVAL;
	}

	length = pdata->oa.length / 2;

	if (pdata->oa.length % 2 == 1) {
		/* There is an extra number in semi-octet and fill bits*/
		length++;
	}

	memcpy(address, buf, length);

	for (int i = 0; i < length; i++) {
		address[i] = swap_nibbles(address[i]);
	}

	convert_number_to_str(address, pdata->oa.length, pdata->oa.address_str);

	/* 2 for length and type fields */
	return 2 + length;
}
